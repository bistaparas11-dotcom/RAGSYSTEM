import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.data.processors.chunking import SECChunker, ChunkType

def test_chunking_with_children_and_summaries():
    # Mock generate_summary to avoid actual LLM calls
    with patch("app.data.processors.chunking.generate_summary") as mock_summary:
        mock_summary.return_value = "Mocked Summary"
        
        chunker = SECChunker(chunk_size=500, overlap=50)
        
        meta = {
            "filing_id":   "TEST10K2024",
            "company_name": "TestCorp",
            "cik":         "0000000000",
            "filing_type": "10-K",
            "filing_date": "2024-01-01",
            "fiscal_year": 2024,
        }
        
        markdown = """
# ITEM 1. BUSINESS
This is some business text that is long enough to be a paragraph. It contains multiple sentences to ensure that the content exceeds the minimum character threshold required for creating a text chunk. We want to make sure that paragraph-based chunking is working correctly and that this text is not discarded as a micro-chunk.

| Year | Revenue |
|------|---------|
| 2023 | $100M   |

# ITEM 1A. RISK FACTORS
We have many risks. These risks include market volatility, regulatory changes, and competitive pressures. Each of these risks could materially affect our business operations and financial results in the upcoming fiscal year.

This is another paragraph of risk factors. It is also sufficiently long to be captured as a separate chunk if it's treated as a paragraph. We are testing the ability of the chunker to split by double newlines.

# ITEM 2. PROPERTIES
We have many properties across the globe. Our main headquarters is located in a high-tech campus that supports our research and development efforts. We also leased several data centers to handle our growing computational needs.
"""
        
        chunks = chunker.chunk(markdown, meta)
        
        print(f"Generated {len(chunks)} chunks.")
        
        # Verify structure
        doc_chunk = [c for c in chunks if c.type == ChunkType.DOCUMENT][0]
        sec_chunks = [c for c in chunks if c.type == ChunkType.SECTION]
        text_chunks = [c for c in chunks if c.type == ChunkType.TEXT]
        table_chunks = [c for c in chunks if c.type == ChunkType.TABLE]
        
        print(f"Doc Chunk ID: {doc_chunk.id}")
        print(f"Doc Child IDs: {doc_chunk.child_chunk_ids}")
        
        assert len(doc_chunk.child_chunk_ids) == len(sec_chunks)
        
        for sec in sec_chunks:
            print(f"Section {sec.section} Child IDs: {sec.child_chunk_ids}")
            # Section children should be text and table chunks belonging to it
            expected_children = [c.id for c in chunks if c.parent_id == sec.id and c.type in [ChunkType.TEXT, ChunkType.TABLE]]
            assert set(sec.child_chunk_ids) == set(expected_children)
            
        for t in text_chunks:
            assert t.summary == "Mocked Summary"
            print(f"Text Chunk {t.id} content:\n{t.content!r}\n")
            
        for tbl in table_chunks:
            assert tbl.summary == "Mocked Summary"
            print(f"Table Chunk {tbl.id} has summary.")

        print("\nVerification successful!")
        
        # Print sample chunks
        if text_chunks:
            print("\nSample Text Chunk Dict:")
            print(json.dumps(text_chunks[0].to_dict(), indent=2))
        
        if table_chunks:
            print("\nSample Table Chunk Dict:")
            print(json.dumps(table_chunks[0].to_dict(), indent=2))

if __name__ == "__main__":
    try:
        test_chunking_with_children_and_summaries()
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)
