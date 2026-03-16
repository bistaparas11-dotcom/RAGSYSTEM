import os
import weaviate
from weaviate.classes.init import Auth

from .chunking import SECChunker
from app.config import settings
from app.utils.file_utils import process_pdf
from app.utils.sec_utils import SECBulkDownloader
from app.utils.metadata_utils import extract_metadata
from app.data.storage.graph_schema import KnowledgeGraph
from app.data.storage.weaviate_schema import WeaviateSchema, WeaviateIngestor

CHUNK_SIZE = 1000
OVERLAP = 100


class IngestionManager:
    def __init__(self):
        self.raw_dir = "./data/raw"
        self.processed_dir = "./data/processed"
        self.prompt = "table"

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

        # -- Chunker ---
        self.chunker = SECChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)

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
        """Iterates through processed Markdowns and chunks them"""
        for filename in os.listdir(self.processed_dir):
            if filename.endswith(".md"):
                md_path = os.path.join(self.processed_dir, filename)
                meta = extract_metadata(md_path)
                chunks = self.chunker.chunk_file(md_path, meta)

    def ingest_all_files(self):
        """
        Chunks every processed Markdown, then writes to both
        Neo4j and Weaviate.
        """
        md_files = [f for f in os.listdir(self.processed_dir) if f.endswith(".md")]
        if not md_files:
            print("No processed Markdown files found.")
            return

        for filename in md_files:
            md_path = os.path.join(self.processed_dir, filename)
            print(f"Processing {filename} ...")

            # 1. Extract metadata and chunk
            meta = extract_metadata(md_path)
            chunks = self.chunker.chunk_file(md_path, meta)

            if not chunks:
                print(f" No chunks produced for {filename}, skipping")
                continue

            # 2. Build Neo4j knowledge graph
            print("Building Neo4j knowledge graph ...")
            self.knowledge_graph.build_graph(chunks, meta)

            # 3. Ingest into Weaviate
            print("Ingesting into Weaviate ...")
            refs = self.weaviate_ingestor.ingest(chunks, meta)
            print(f" Weaviate: {len(refs)} objects upserted.")

        print("\nIngestion complete.")

    def close(self):
        """Clean up connections."""
        self.knowledge_graph.close()
        self.weaviate_client.close()


if __name__ == "__main__":
    filing_10k_url_meta = "https://www.sec.gov/ix?doc=/Archives/edgar/data/0001326801/000162828026003942/meta-20251231.htm"
    filing_10k_url_nvda = "https://www.sec.gov/ix?doc=/Archives/edgar/data/0001045810/000104581026000021/nvda-20260125.htm"
    companies = [filing_10k_url_meta, filing_10k_url_nvda]

    manager = IngestionManager()

    try:
        # Download SEC files in PDF
        # print("---------- Downloading SEC files in PDF ----------")
        # manager.download_files(companies, 2025)

        # Process PDFs to Markdown
        print("---------- Processing PDFs to Markdown ----------")
        manager.process_all_files()

        # Chunk + build graph + insert into Weaviate
        print("---------- Chunking + Building Graph + Inserting into Weaviate ----------")
        manager.ingest_all_files()

    finally:
        manager.close()

