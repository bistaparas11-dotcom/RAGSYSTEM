import os
from sec_api import QueryApi, PdfGeneratorApi
from ..config import settings


class SECBulkDownloader:
    def __init__(self):
        self.query_api = QueryApi(api_key=settings.SEC_API_KEY)
        self.pdf_api = PdfGeneratorApi(api_key=settings.SEC_API_KEY)
        self.save_path = "./data/raw"

    def download_10ks(self, tickers: list, year: int):
        for ticker in tickers:
            print(f"Searching for {ticker} 10-K in {year}...")
            
            # Build the Query
            query_str = (
                f'ticker:{ticker} xAND '
                f'formType:"10-K" AND '
                f'filedAt:[{year}-01-01 TO {year}-12-31]'
            )
            
            response = self.query_api.get_filings({
                "query": query_str,
                "from": "0",
                "size": "1", # Usually only one 10-K per year
                "sort": [{"filedAt": {"order": "desc"}}]
            })

            # Extract URL and Download PDF
            filings = response.get('filings', [])
            if filings:
                filing_url = filings[0]['linkToFilingDetails']
                filename = f"{ticker}_10K_{year}.pdf"
                output_file = os.path.join(self.save_path, filename)
                
                print(f"Generating PDF for {ticker}...")
                pdf_content = self.pdf_api.get_pdf(filing_url)
                
                with open(output_file, "wb") as f:
                    f.write(pdf_content)
                print(f"Successfully saved to {output_file}")
            else:
                print(f"No 10-K found for {ticker} in {year}.")


if __name__ == "__main__":
    downloader = SECBulkDownloader()
    # Example usage
    companies = ["AAPL", "GOOGL"]
    downloader.download_10ks(companies, 2024)