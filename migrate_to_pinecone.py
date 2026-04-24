"""
Migration Script: Weaviate to Pinecone

This script re-ingests all cached enriched chunks into new Pinecone index.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from app.data.processors.ingestion_manager import IngestionManager

def run_migration():
    print("\n Starting Migration to Pinecone...")
    
    manager = IngestionManager()
    
    try:
        manager.ingest_pinecone_only(force=True)
        print("\n Migration to Pinecone completed successfully!")
    except Exception as e:
        print(f"\n Migration failed: {e}")
    finally:
        manager.close()

if __name__ == "__main__":
    run_migration()
