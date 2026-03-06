from pathlib import Path


company_names = {
    "MSFT": "Microsoft",
    "GOOG": "Google (Class C)",
    "GOOGL": "Google (Class A)",
    "META": "Meta",
    "NVDA": "NVIDIA",
    "ORCL": "Oracle"
}

def extract_metadata(md_path: str) -> dict:
    # Parse the filename to extract metadata
    filename = Path(md_path).name
    ticker, filing_type, year = filename.split("_")

    metadata = {
        "filing_id": f"{ticker}{filing_type}{year}",
        "company_name": company_names.get(ticker, ticker),
        "filing_type": filing_type,
        "fiscal_year": year
    }
    
    return metadata
