import os
from ...utils.fetch_sec import SECBulkDownloader
from ...utils.pdf_to_markdown import process_pdf


class IngestionManager:
    def __init__(self):
        self.raw_dir = "./data/raw"
        self.processed_dir = "./data/processed"
        self.prompt = "table"
    
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


if __name__ == "__main__":
    companies = ["NVDA", "ORCL", "META", "MSFT", "GOOGL"]
    manager = IngestionManager()
    # manager.download_files(companies, 2024)
    manager.download_files(companies, 2025)
    manager.process_all_files()
