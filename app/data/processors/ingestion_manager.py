"""
Ingestion Manager

Run everything at once:
    manager.run_full_pipeline()

Run each step separately:
    manager.chunk_all_files()
    manager.enrich_all_files()
    manager.ingest_all_files()

Check status:
    manager.status()

Force redo a single stage for one file
    manager.reset_file("meta-20251231.md", stage="weaviate")

Force redo a stage for all files:
    manager.reset_all(stage="neo4j")
"""

import os
import weaviate
from weaviate.classes.init import Auth

from app.config import settings
from app.utils.file_utils import process_pdf
from app.utils.sec_utils import SECBulkDownloader
from app.utils.metadata_utils import extract_metadata
from app.data.processors.chunk_cache import ChunkCache
from app.data.processors.fast_chunk_pipeline import FastChunkPipeline
from app.data.storage.graph_schema import KnowledgeGraph
from app.data.storage.weaviate_schema import WeaviateSchema, WeaviateIngestor


class IngestionManager:
    def __init__(self):
        self.raw_dir = "./data/raw"
        self.processed_dir = "./data/processed"
        self.prompt = "table"
        self.cache = ChunkCache("./data/chunks")
        self.pipeline = FastChunkPipeline()

        # --- Neo4j ---
        self.knowledge_graph = KnowledgeGraph(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )

        # --- Weaviate ---
        self.weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.WEAVIATE_URL,
            auth_credentials=Auth.api_key(settings.WEAVIATE_API_KEY),
        )
        WeaviateSchema(self.weaviate_client).create_schema()
        self.weaviate_ingestor = WeaviateIngestor(
            client=self.weaviate_client
        )

    def close(self):
        self.knowledge_graph.close()
        self.weaviate_client.close()

    def download_files(self, ticker_urls, year):
        downloader = SECBulkDownloader()
        downloader.download_10ks(ticker_urls, year)

    def process_all_files(self):
        """Iterates through raw PDFs and saves Markdowns"""
        for filename in os.listdir(self.raw_dir):
            if filename.endswith(".pdf"):
                raw_path = os.path.join(self.raw_dir, filename)
                output_path = os.path.join(
                    self.processed_dir, filename.replace(".pdf", ".md"))

                # Check if already processed to save Modal credits
                if not os.path.exists(output_path):
                    print(f"Processing {filename} via PaddleOCR-VL...")
                    process_pdf(pdf_path=raw_path,
                                output_md_path=output_path,
                                prompt=self.prompt)
                else:
                    print(f"Skipping {filename}, already processed.")

    def chunk_all_files(self):
        """Pure structural chunking - no LLM."""
        self.pipeline._chunk_all()

    def enrich_all_files(self):
        """Enrichment using Deepseek-R1."""
        self.pipeline.enrich_all()

    def ingest_all_files(self):
        """Load cached enriched chunks, then writes to both Neo4j and Weaviate."""
        cached_files = self.cache.list_cached()
        if not cached_files:
            print("No cached chunks found.")
            return

        for cache_file in sorted(cached_files):
            print(f"Ingesting {cache_file} ...")

            result = self.cache.load(cache_file)
            if not result:
                print(f" [!] Could not load, skipping.")
                continue
            chunks, meta = result

            if self.cache.is_done(cache_file, "neo4j"):
                print(f" - Neo4j ingestion already done, skipping.")
            else:
                try:
                    print(" - Building Neo4j knowledge graph...")
                    self.knowledge_graph.build_graph(chunks, meta)
                    self.cache.mark(cache_file, "neo4j")
                    print(" [OK] Neo4j done.")
                except Exception as e:
                    print(f" [X] Failed to build Neo4j graph: {e}")

            if self.cache.is_done(cache_file, "weaviate"):
                print(f" - Weaviate ingestion already done, skipping.")
            else:
                try:
                    print(" - Ingesting into Weaviate...")
                    refs = self.weaviate_ingestor.ingest(chunks, meta)
                    self.cache.mark(cache_file, "weaviate")
                    print(f" [OK] Weaviate done: {len(refs)} objects upserted.")
                except Exception as e:
                    print(f" [X] Failed to ingest into Weaviate: {e}")

        print("\nIngestion complete.")
        self.cache.print_status()

    def run_full_pipeline(self):
        print("\n---------- Running Full Pipeline ----------")
        print("STAGE 1/3 - Structural Chunking (no LLM)")
        self.chunk_all_files()
        print("STAGE 2/3 - Enrichment via Modal Deepseek-R1")
        self.enrich_all_files()
        print("STAGE 3/3 - Ingestion (Neo4j + Weaviate)")
        self.ingest_all_files()
        print("\n[OK] Full pipeline completed.")

    def status(self):
        self.pipeline.status()

    def reset_file(self, cache_file: str, stage: str = None):
        self.cache.reset(cache_file, stage)
        print(f"[OK] Reset {cache_file} for stage: {stage}")

    def reset_all(self, stage: str = None):
        for f in self.cache.list_cached():
            self.cache.reset(f, stage)
        print(f"[OK] Reset all files for stage: {stage}")


if __name__ == "__main__":
    filing_10k_url_meta = "https://www.sec.gov/ix?doc=/Archives/edgar/data/0001326801/000162828026003942/meta-20251231.htm"
    filing_10k_url_nvda = "https://www.sec.gov/ix?doc=/Archives/edgar/data/0001045810/000104581026000021/nvda-20260125.htm"
    companies = [filing_10k_url_meta, filing_10k_url_nvda]

    manager = IngestionManager()

    try:
        # manager.download_files(companies, 2025)
        # manager.process_all_files()
        manager.run_full_pipeline()
        # manager.status()
    finally:
        manager.close()

