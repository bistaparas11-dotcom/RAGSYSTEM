import os

from chunking import SECChunker
from graph_schema import KnowledgeGraph
from app.config import settings
from app.utils.sec_utils import SECBulkDownloader
from app.utils.file_utils import process_pdf


class IngestionManager:
    def __init__(self):
        self.raw_dir = "./data/raw"
        self.processed_dir = "./data/processed"
        self.prompt = "table"
        self.knowledge_graph = KnowledgeGraph(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD
        )
    
    def download_files(self, tickers, year):
        downloader = SECBulkDownloader()
        downloader.download_10ks(tickers, year)

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
        pass

    def build_graph(self):
        """Builds the knowledge graph from the chunks"""
        pass
        

if __name__ == "__main__":
    # companies = ["GOOG", "GOOGL", "META", "MSFT", "NVDA", "ORCL"]
    manager = IngestionManager()

    # Download SEC files in PDF
    # manager.download_files(companies, 2024)

    # Process PDFs to Markdown
    # manager.process_all_files()

    # Chunk the Markdown files
    manager.chunk_all_files()

    # Build the knowledge graph
    manager.build_graph()
    