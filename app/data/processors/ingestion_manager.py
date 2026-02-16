import os
from utils.pdf_to_markdown import process_pdf


class IngestionManager:
    def __init__(self):
        self.raw_dir = "data/storage/raw"
        self.processed_dir = "data/storage/processed"
        self.prompt = "table"

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
                                output_md_path=output_path, prompt=self.prompt)
                else:
                    print(f"Skipping {filename}, already processed.")


if __name__ == "__main__":
    manager = IngestionManager()
    manager.process_all_files()
