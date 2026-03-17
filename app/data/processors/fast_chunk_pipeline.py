import os
import re
import json
import httpx
import asyncio
from typing import List, Dict, Any, Optional

from app.config import settings
from app.utils.llm_utils import ENRICH_PROMPT
from app.utils.metadata_utils import extract_metadata
from app.data.processors.chunk_cache import ChunkCache
from app.data.processors.chunking import SECChunker, Chunk, ChunkType


PROCESSED_DIR  = "./data/processed"
CHUNKS_DIR     = "./data/chunks"
ENRICHMENT_DIR = "./data/enrichments"

BATCH_SIZE   = 10    # concurrent requests per batch
MAX_RETRIES  = 3
RETRY_DELAY  = 5     # seconds on failure
HTTP_TIMEOUT = 120   # Modal cold-start can be slow


class FastChunkPipeline:

    def __init__(self):
        self.chunker = SECChunker(chunk_size=1500, overlap=150)
        self.cache = ChunkCache(CHUNKS_DIR)
        self.modal_url = settings.DEEPSEEK_API_URL
        os.makedirs(ENRICHMENT_DIR, exist_ok=True)

    def run_all(self):
        """Chunk then enrich all files."""
        self._chunk_all()
        asyncio.run(self._enrich_all())

    def enrich_all(self):
        """Enrich already-chunked files (resumable)."""
        asyncio.run(self._enrich_all())

    def status(self):
        self.cache.print_status()
        cached = self.cache.list_cached()
        print(f"\nEnrichment progress:")
        for fname in sorted(cached):
            result = self.cache.load(fname)
            if not result:
                continue
            chunks, _ = result
            enrichable = [c for c in chunks if c.type in (ChunkType.TEXT, ChunkType.TABLE)]
            done = self._load_enrichments(fname)
            done_count = sum(1 for c in enrichable if c.id in done)
            print(f"  {fname:<40} {done_count}/{len(enrichable)} enriched")
        print()

    # --------------- Stage 1: Structural chunking (no LLM) --------------- #
    def _chunk_all(self):
        md_files = sorted(f for f in os.listdir(PROCESSED_DIR) if f.endswith(".md"))
        if not md_files:
            print("No .md files found in", PROCESSED_DIR)
            return

        for filename in md_files:   
            if self.cache.is_done(filename, "chunked"):
                print(f" - Already chunked: {filename}")
                continue

            md_path = os.path.join(PROCESSED_DIR, filename)
            print(f"\n----- Chunking {filename} -----")

            try:
                meta = extract_metadata(md_path)
                chunks = self.chunker.chunk_file(md_path, meta, skip_enrichment=True)

                if not chunks:
                    print(f" [!] No chunks produced.")
                    continue

                self.cache.save(filename, chunks, meta)
                self.cache.mark(filename, "chunked")

                enrichable = [c for c in chunks if c.type in (ChunkType.TEXT, ChunkType.TABLE)]
                print(f" [OK] {len(chunks)} chunks ({len(enrichable)} to enrich) saved.")

            except KeyboardInterrupt:
                print(f"\n [!] Interrupted. Completed files are safe.")
                break
            except Exception as e:
                print(f" [X] Failed: {e}")
                continue

    # --------------- Stage 2: Async enrichment via Modal DeepSeek-R1 --------------- #
    async def _enrich_all(self):
        for fname in sorted(self.cache.list_cached()):
            if not self.cache.is_done(fname, "chunked"):
                continue
            await self._enrich_file(fname)

    async def _enrich_file(self, cache_file: str):
        result = self.cache.load(cache_file)
        if not result:
            return
        chunks, meta = result

        enrichable = [c for c in chunks if c.type in (ChunkType.TEXT, ChunkType.TABLE)]
        done = self._load_enrichments(cache_file)
        todo = [c for c in enrichable if c.id not in done]

        if not todo:
            print(f" - All enrichments done for {cache_file}")
            self._apply_enrichments(chunks, done)
            self.cache.save(cache_file, chunks, meta)
            return

        total = len(enrichable)
        print(f"\n----- Enriching {cache_file}: {len(todo)} remaining / {total} total -----")

        # Single shared httpx client for all batches in this file
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            for i in range(0, len(todo), BATCH_SIZE):
                batch = todo[i:i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                total_batches = (len(todo) - 1) // BATCH_SIZE + 1
                print(
                    f"  Batch {batch_num}/{total_batches} "
                    f"({len(batch)} chunks)...",
                    end=" ", flush=True,
                )

                results = await self._batch_enrich(client, batch)

                for chunk, enrichment in zip(batch, results):
                    if enrichment:
                        done[chunk.id] = enrichment

                # Checkpoint after every batch
                self._save_enrichments(cache_file, done)
                done_total = sum(1 for c in enrichable if c.id in done)
                print(f"[OK] ({done_total}/{total} done)")
                await asyncio.sleep(0.5)

        # Write enrichments into chunks and re-save cache
        self._apply_enrichments(chunks, done)
        self.cache.save(cache_file, chunks, meta)
        print(f"[OK] Enrichment complete for {cache_file}")

    async def _batch_enrich(
        self, client: httpx.AsyncClient, chunks: List[Chunk]
    ) -> List[Optional[Dict]]:
        tasks = [self._enrich_one(client, c) for c in chunks]
        return await asyncio.gather(*tasks)

    async def _enrich_one(
        self, client: httpx.AsyncClient, chunk: Chunk
    ) -> Optional[Dict[str, Any]]:
        """Single async POST to Modal DeepSeek-R1 endpoint."""
        prompt = ENRICH_PROMPT.format(
            section=chunk.section,
            content=chunk.content[:1500],
        )
        payload = {"prompt": prompt, "stream": False}

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    self.modal_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()

                response_text = response.json()["response"]
                return self._parse_enrichment(response_text)

            except httpx.TimeoutException:
                print(f"\n [!] Timeout on chunk {chunk.id} (attempt {attempt+1}/{MAX_RETRIES})...",
                      end=" ", flush=True)
                await asyncio.sleep(RETRY_DELAY)

            except httpx.HTTPStatusError as e:
                # Modal 429 = too many requests
                if e.response.status_code == 429:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"\n [!] Modal rate limit — waiting {wait}s...", end=" ", flush=True)
                    await asyncio.sleep(wait)
                else:
                    print(f"\n [X] HTTP {e.response.status_code} for chunk {chunk.id}: {e}")
                    return None

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"\n [X] Failed chunk {chunk.id} after {MAX_RETRIES} attempts: {e}")
                    return None
                await asyncio.sleep(RETRY_DELAY)

        return None

    @staticmethod
    def _parse_enrichment(response_text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from DeepSeek response."""
        # Strip DeepSeek-R1 chain-of-thought block
        text = re.sub(r"<think>[\s\S]*?</think>", "", response_text).strip()

        # Find the JSON object
        json_match = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
        if not json_match:
            return {"summary": text[:300], "entities": [], "relationships": []}

        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return {"summary": text[:300], "entities": [], "relationships": []}

    def _enrichment_path(self, cache_file: str) -> str:
        key = cache_file.replace(".json", "").replace(".md", "")
        return os.path.join(ENRICHMENT_DIR, f"{key}_enrichments.json")

    def _load_enrichments(self, cache_file: str) -> Dict[str, Dict]:
        path = self._enrichment_path(cache_file)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_enrichments(self, cache_file: str, enrichments: Dict[str, Dict]):
        path = self._enrichment_path(cache_file)
        with open(path, "w") as f:
            json.dump(enrichments, f, indent=2)

    @staticmethod
    def _apply_enrichments(chunks: List[Chunk], enrichments: Dict[str, Dict]):
        for chunk in chunks:
            e = enrichments.get(chunk.id)
            if e:
                chunk.summary       = e.get("summary")
                chunk.entities      = e.get("entities", [])
                chunk.relationships = e.get("relationships", [])
