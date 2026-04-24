import uuid
from typing import Dict, Any, List, Optional
from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.utils.embedding_utils import BGEM3Embedder


class PineconeIngestor:
    """
    Inserts chunks into Pinecone with BGE-M3 vectors.
    Since Pinecone is a flat vector store, we use metadata to distinguish 
    between Document, Section, Chunk, and Entity types.
    """

    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024, # BGE-M3
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION
                )
            )
            
        self.index = self.pc.Index(self.index_name)
        self.embedder = BGEM3Embedder()
        self._refs: Dict[str, str] = {}

    def ingest(self, chunks, filing_metadata: Dict[str, Any]) -> Dict[str, str]:
        from app.data.processors.chunking import ChunkType

        filing_id = filing_metadata["filing_id"]
        
        doc_chunk = next((c for c in chunks if c.type == ChunkType.DOCUMENT), None)
        if doc_chunk:
            self._insert_document(doc_chunk, filing_metadata)

        for chunk in chunks:
            if chunk.type == ChunkType.SECTION:
                self._insert_section(chunk, filing_id)

        for chunk in chunks:
            if chunk.type in (ChunkType.TEXT, ChunkType.TABLE):
                self._insert_chunk(chunk, filing_id)
                for entity in chunk.entities:
                    self._insert_entity(entity, filing_id)
                # Note: Pinecone doesn't natively support cross-references like Weaviate's reference_add.
                # Relationships between entities are handled in Neo4j.
                # We store IDs in metadata for retrieval convenience.

        return self._refs

    def _insert_document(self, chunk, metadata: Dict[str, Any]):
        filing_id = metadata["filing_id"]
        vector = self.embedder.embed(chunk.content)
        
        # Pinecone ID must be a string
        pid = f"doc_{filing_id}"
        
        self.index.upsert(
            vectors=[{
                "id": pid,
                "values": vector,
                "metadata": {
                    "type": "Document",
                    "neo4j_id": filing_id,
                    "filing_id": filing_id,
                    "company_name": metadata.get("company_name", ""),
                    "filing_type": metadata.get("filing_type", ""),
                    "fiscal_year": metadata.get("fiscal_year", ""),
                    "content": chunk.content[:20000] # Pinecone metadata limit is 40KB, being safe
                }
            }]
        )
        self._refs[filing_id] = pid

    def _insert_section(self, chunk, filing_id: str):
        vector = self.embedder.embed(chunk.content)
        pid = f"sec_{chunk.id}"
        
        self.index.upsert(
            vectors=[{
                "id": pid,
                "values": vector,
                "metadata": {
                    "type": "Section",
                    "neo4j_id": chunk.id,
                    "chunk_id": chunk.id,
                    "filing_id": filing_id,
                    "section": chunk.section,
                    "content": chunk.content[:20000],
                    "is_risk": chunk.metadata.get("is_risk", False),
                    "has_tables": chunk.metadata.get("has_tables", False),
                }
            }]
        )
        self._refs[chunk.id] = pid

    def _insert_chunk(self, chunk, filing_id: str):
        vector = self.embedder.embed(chunk.summary or chunk.content[:512])
        pid = f"chk_{chunk.id}"
        
        self.index.upsert(
            vectors=[{
                "id": pid,
                "values": vector,
                "metadata": {
                    "type": "Chunk",
                    "neo4j_id": chunk.id,
                    "chunk_id": chunk.id,
                    "filing_id": filing_id,
                    "section": chunk.section,
                    "chunk_type": chunk.type.value,
                    "is_financial": chunk.metadata.get("is_financial", False),
                    "content": chunk.content[:5000],
                    "summary": chunk.summary or "",
                    "parent_id": chunk.parent_id or ""
                }
            }]
        )
        self._refs[chunk.id] = pid

    def _insert_entity(self, entity: Dict[str, Any], filing_id: str):
        name = entity.get('name', 'Unknown')
        etype = entity.get('type', 'Unknown')
        neo4j_id = f"{name}||{etype}"
        
        if neo4j_id in self._refs:
            return
            
        vector = self.embedder.embed(entity.get("description") or name)
        pid = f"ent_{uuid.uuid5(uuid.NAMESPACE_DNS, neo4j_id)}"
        
        self.index.upsert(
            vectors=[{
                "id": pid,
                "values": vector,
                "metadata": {
                    "type": "Entity",
                    "neo4j_id": neo4j_id,
                    "name": name,
                    "entity_type": etype,
                    "filing_id": filing_id,
                    "description": entity.get("description", "")
                }
            }]
        )
        self._refs[neo4j_id] = pid
