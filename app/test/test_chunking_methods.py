import json
from ..data.processors.chunking import SECHierarchicalChunker, ContentType


def test_chunking():
    chunker = SECHierarchicalChunker()

    markdown_text = """
ITEM 1. BUSINESS
Apple Inc. is a global leader in technology.
| Product | Sales |
|---------|-------|
| iPhone  | $200B |
| Mac     | $40B  |

ITEM 1A. RISK FACTORS
We face competition.
"""

    filing_metadata = {
        "accession_number": "0000320193-23-000106",
        "company_name": "Apple Inc."
    }

    chunks = chunker.parse_markdown(markdown_text, filing_metadata)

    print(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(f"ID: {chunk.id}")
        print(f"Type: {chunk.content_type}")
        print(f"Section: {chunk.section}")
        print(f"Content Length: {len(chunk.content)}")
        if chunk.entities:
            print(f"Entities: {chunk.entities}")
        if chunk.metadata:
            print(f"Metadata keys: {list(chunk.metadata.keys())}")


if __name__ == "__main__":
    test_chunking()
