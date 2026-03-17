import os
import json
from dataclasses import asdict
from typing import List, Optional, Dict, Any

from app.data.processors.chunking import Chunk, ChunkType


CHUNKS_DIR = "./data/chunks"
STATE_FILE = os.path.join(CHUNKS_DIR, "_ingestion_state.json")


def _chunk_to_dict(chunk: Chunk) -> Dict[str, Any]:
    return {
        "id": chunk.id,
        "type": chunk.type.value,
        "section": chunk.section,
        "content": chunk.content,
        "summary": chunk.summary,
        "parent_id": chunk.parent_id,
        "child_chunk_ids": chunk.child_chunk_ids,
        "entities": chunk.entities,
        "relationships": chunk.relationships,
        "metadata": chunk.metadata,
    }

def _dict_to_chunk(d: Dict[str, Any]) -> Chunk:
    chunk = Chunk(
        id = d["id"],
        type = ChunkType(d["type"]),
        section = d.get("section", ""),
        content = d.get("content", ""),
        summary = d.get("summary"),
        parent_id = d.get("parent_id"),
        child_chunk_ids = d.get("child_chunk_ids", []),
        entities = d.get("entities", []),
        relationships = d.get("relationships", []),
        metadata = d.get("metadata", {}),
    )
    return chunk


class ChunkCache:
    """Read/write chunk lists to disk and also tracks ingestion progress."""

    def __init__(self, cache_dir: str = CHUNKS_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._state = self._load_state()

    def save(self, filename: str, chunks: List[Chunk], metadata: Dict[str, Any]):
        """Serialise chunks to <cache_dir>/<filename>.json"""
        path = self._path(filename)
        payload = {
            "metadata": metadata,
            "chunks": [_chunk_to_dict(c) for c in chunks]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f" [Cached] {len(chunks)} chunks -> {path}")

    def load(self, filename: str) -> Optional[tuple[List[Chunk], Dict[str, Any]]]:
        """Load chunks from disk.
        Returns:
            (chunks, metadata) or None if not cached."""
        path = self._path(filename)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        chunks = [_dict_to_chunk(d) for d in payload["chunks"]]
        metadata = payload["metadata"]
        print(f" [Loaded] {len(chunks)} chunks -> {path}")
        return chunks, metadata

    def exists(self, filename: str) -> bool:
        return os.path.exists(self._path(filename))

    def list_cached(self) -> List[str]:
        """Return all cached chunk filenames."""
        return [
            f for f in os.listdir(self.cache_dir)
            if f.endswith(".json") and not f.startswith("_")
        ]

    def delete(self, filename: str):
        path = self._path(filename)
        if os.path.exists(path):
            os.remove(path)

    def mark(self, filename: str, stage: str, done: bool = True):
        """Mark a stage as complete for a file.
            stage: "chunked" | "neo4j" | "weaviate"
        """
        key = self._state_key(filename)
        if key not in self._state:
            self._state[key] = {"chunked": False, "neo4j": False, "weaviate": False}
        self._state[key][stage] = done
        self._save_state()

    def is_done(self, filename: str, stage: str) -> bool:
        key = self._state_key(filename)
        return self._state.get(key, {}).get(stage, False)

    def reset(self, filename: str, stage: Optional[str] = None):
        """Reset state for a file."""
        key = self._state_key(filename)
        if stage:
            if key in self._state:
                self._state[key][stage] = False
        else:
            self._state.pop(key, None)
        self._save_state()

    def print_status(self):
        """Print ingestion status for all cached files."""
        cached = self.list_cached()
        if not cached:
            print("No cached files found.")
            return
        print(f"\n{'File':<35} {'Chunked':<10} {'Neo4j':<10} {'Weaviate'}")
        print(f"{'─'*35} {'─'*10} {'─'*10} {'─'*8}")
        for fname in sorted(cached):
            key = self._state_key(fname)
            state = self._state.get(key, {})
            c = "[OK]" if state.get("chunked") else "[X]"
            n = "[OK]" if state.get("neo4j") else "[X]"
            w = "[OK]" if state.get("weaviate") else "[X]"
            print(f"{fname:<35} {c:<10} {n:<10} {w}")
        print()

    def _path(self, filename: str) -> str:
        if not filename.endswith(".json"):
            filename = filename.replace(".md", "") + ".json"
        return os.path.join(self.cache_dir, filename)

    @staticmethod
    def _state_key(filename: str) -> str:
        return filename.replace(".md", "").replace(".json", "")

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self._state, f, indent=2)
