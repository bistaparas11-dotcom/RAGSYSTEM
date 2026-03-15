import os
from sec_api import PdfGeneratorApi
from app.config import settings


class SECBulkDownloader:
    def __init__(self):
        self.pdf_api = PdfGeneratorApi(api_key=settings.SEC_API_KEY)
        self.save_path = "./data/raw"

    def download_10ks(self, ticker_urls: list, year: int):
        for filing_url in ticker_urls:
            pdf_10k_filing = self.pdf_api.get_pdf(filing_url)
            ticker = filing_url.split('/')[-1].split(('-'))[0]
            filename = f"{ticker}_10K_{year}.pdf"
            output_file = os.path.join(self.save_path, filename)

            with open(output_file, "wb") as file:
                file.write(pdf_10k_filing)

            print(f"Successfully saved to {output_file}")


if __name__ == "__main__":
    downloader = SECBulkDownloader()
    # Example usage
    filing_10k_url_meta = "https://www.sec.gov/ix?doc=/Archives/edgar/data/0001326801/000162828026003942/meta-20251231.htm"
    filing_10k_url_nvda = "https://www.sec.gov/ix?doc=/Archives/edgar/data/0001045810/000104581026000021/nvda-20260125.htm"
    companies = [filing_10k_url_meta, filing_10k_url_nvda]
    downloader.download_10ks(companies, 2025)