import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    ReferenceProperty,
    VectorDistances,
)
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from typing import Dict, Any, List, Optional

from app.utils.embedding_utils import BGEM3Embedder


class WeaviateSchema:
    """
    Create schema for Weaviate database corresponding to Neo4j nodes.
    """

    COLLECTIONS = ["Document", "Section", "Chunk", "Entity"]

    def __init__(self, client: weaviate.WeaviateClient):
        self.client = client

    def create_schema(self):
        self._create_document_collection()
        self._create_entity_collection()
        self._create_section_collection()
        self._create_chunk_collection()

    def drop_schema(self):
        for name in reversed(self.COLLECTIONS):
            if self.client.collections.exists(name):
                self.client.collections.delete(name)

    # shared HNSW config (Hierarchical Navigable Small World)
    @staticmethod
    def _hnsw(ef: int = 128):
        return Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=ef,
            max_connections=64,
            ef=ef,
        )

    def _create_document_collection(self):
        if self.client.collections.exists("Document"):
            return
        self.client.collections.create(
            name="Document",
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=self._hnsw(),
            properties=[
                Property(name="neo4j_id",     data_type=DataType.TEXT, index_filterable=True, index_searchable=False),
                Property(name="filing_id",    data_type=DataType.TEXT, index_filterable=True),
                Property(name="company_name", data_type=DataType.TEXT),
                Property(name="filing_type",  data_type=DataType.TEXT, index_filterable=True),
                Property(name="fiscal_year",  data_type=DataType.TEXT, index_filterable=True),
                Property(name="content",      data_type=DataType.TEXT),
            ],
        )

    def _create_section_collection(self):
        if self.client.collections.exists("Section"):
            return
        self.client.collections.create(
            name="Section",
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=self._hnsw(),
            properties=[
                Property(name="neo4j_id",  data_type=DataType.TEXT, index_filterable=True, index_searchable=False),
                Property(name="chunk_id",  data_type=DataType.TEXT, index_filterable=True),
                Property(name="filing_id", data_type=DataType.TEXT, index_filterable=True),
                Property(name="section",   data_type=DataType.TEXT, index_filterable=True),
                Property(name="content",   data_type=DataType.TEXT),
                Property(name="is_risk",   data_type=DataType.BOOL, index_filterable=True),
                Property(name="has_tables",data_type=DataType.BOOL, index_filterable=True),
            ],
            references=[
                ReferenceProperty(name="partOfDocument", target_collection="Document"),
            ],
        )

    def _create_chunk_collection(self):
        if self.client.collections.exists("Chunk"):
            return
        self.client.collections.create(
            name="Chunk",
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=256,
                max_connections=64,
            ),
            properties=[
                Property(name="neo4j_id",    data_type=DataType.TEXT, index_filterable=True, index_searchable=False),
                Property(name="chunk_id",    data_type=DataType.TEXT, index_filterable=True),
                Property(name="filing_id",   data_type=DataType.TEXT, index_filterable=True),
                Property(name="section",     data_type=DataType.TEXT, index_filterable=True),
                Property(name="chunk_type",  data_type=DataType.TEXT, index_filterable=True),
                Property(name="is_financial",data_type=DataType.BOOL, index_filterable=True),
                Property(name="content",     data_type=DataType.TEXT),
                Property(name="summary",     data_type=DataType.TEXT),  # embedded field
            ],
            references=[
                ReferenceProperty(name="partOfSection",    target_collection="Section"),
                ReferenceProperty(name="partOfDocument",   target_collection="Document"),
                ReferenceProperty(name="mentionsEntities", target_collection="Entity"),
            ],
        )

    def _create_entity_collection(self):
        if self.client.collections.exists("Entity"):
            return
        self.client.collections.create(
            name="Entity",
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=self._hnsw(),
            properties=[
                Property(name="neo4j_id",        data_type=DataType.TEXT,   index_filterable=True, index_searchable=False),
                Property(name="name",            data_type=DataType.TEXT,   index_filterable=True, index_searchable=True),
                Property(name="entity_type",     data_type=DataType.TEXT,   index_filterable=True),
                Property(name="filing_id",       data_type=DataType.TEXT,   index_filterable=True),
                Property(name="description",     data_type=DataType.TEXT),  # embedded field
            ],
            references=[
                ReferenceProperty(name="belongsToDocument", target_collection="Document"),
                ReferenceProperty(name="relatesTo",         target_collection="Entity"),
            ],
        )


# ------------------- Ingestor ------------------- #

class WeaviateIngestor:
    """
    Inserts chunks into Weaviate with BGE-M3 vectors.
    Mirrors KnowledgeGraph.build_graph() pass order.

    Usage:
        ingestor = WeaviateIngestor(weaviate_client, nvidia_api_key)
        refs = ingestor.ingest(chunks, filing_metadata)
    """

    def __init__(self, client: weaviate.WeaviateClient, nvidia_api_key: str):
        self.client   = client
        self.embedder = BGEM3Embedder(nvidia_api_key)
        # neo4j_id -> weaviate UUID
        self._refs: Dict[str, str] = {}

    def ingest(self, chunks, filing_metadata: Dict[str, Any]) -> Dict[str, str]:
        from app.data.processors.chunking import ChunkType

        # Pass 1 — insert objects with vectors
        doc_chunk = next((c for c in chunks if c.type == ChunkType.DOCUMENT), None)
        if doc_chunk:
            self._insert_document(doc_chunk, filing_metadata)

        for chunk in chunks:
            if chunk.type == ChunkType.SECTION:
                self._insert_section(chunk, filing_metadata["filing_id"])

        for chunk in chunks:
            if chunk.type in (ChunkType.TEXT, ChunkType.TABLE):
                self._insert_chunk(chunk, filing_metadata["filing_id"])
                for entity in chunk.entities:
                    self._insert_entity(entity, filing_metadata["filing_id"])
                for rel in chunk.relationships:
                    self._add_entity_relation(rel)

        # Pass 2 — wire cross-references
        self._wire_references(chunks, filing_metadata["filing_id"])

        return self._refs

    def _insert_document(self, chunk, metadata: Dict[str, Any]):
        filing_id = metadata["filing_id"]
        vector    = self.embedder.embed(chunk.content)
        col       = self.client.collections.get("Document")
        uuid = col.data.insert(
            properties={
                "neo4j_id":    filing_id,
                "filing_id":   filing_id,
                "company_name":metadata.get("company_name"),
                "filing_type": metadata.get("filing_type"),
                "fiscal_year": metadata.get("fiscal_year"),
                "content":     chunk.content,
            },
            vector=vector,
        )
        self._refs[filing_id] = str(uuid)

    def _insert_section(self, chunk, filing_id: str):
        vector = self.embedder.embed(chunk.content)
        col    = self.client.collections.get("Section")
        uuid = col.data.insert(
            properties={
                "neo4j_id":  chunk.id,
                "chunk_id":  chunk.id,
                "filing_id": filing_id,
                "section":   chunk.section,
                "content":   chunk.content,
                "is_risk":   chunk.metadata.get("is_risk", False),
                "has_tables":chunk.metadata.get("has_tables", False),
            },
            vector=vector,
        )
        self._refs[chunk.id] = str(uuid)

    def _insert_chunk(self, chunk, filing_id: str):
        # Embed summary — cleaner signal than raw text
        vector = self.embedder.embed(chunk.summary or chunk.content[:512])
        col    = self.client.collections.get("Chunk")
        uuid = col.data.insert(
            properties={
                "neo4j_id":    chunk.id,
                "chunk_id":    chunk.id,
                "filing_id":   filing_id,
                "section":     chunk.section,
                "chunk_type":  chunk.type.value,
                "is_financial":chunk.metadata.get("is_financial", False),
                "content":     chunk.content[:2000],
                "summary":     chunk.summary,
            },
            vector=vector,
        )
        self._refs[chunk.id] = str(uuid)

    def _insert_entity(self, entity: Dict[str, Any], filing_id: str):
        neo4j_id = f"{entity.get('name', 'Unknown')}||{entity.get('type', 'Unknown')}"
        if neo4j_id in self._refs:
            return
        vector = self.embedder.embed(entity.get("description") or entity.get("name", ""))
        col    = self.client.collections.get("Entity")
        uuid = col.data.insert(
            properties={
                "neo4j_id":        neo4j_id,
                "name":            entity.get("name", "Unknown"),
                "entity_type":     entity.get("type", "Unknown"),
                "filing_id":       filing_id,
                "description":     entity.get("description", ""),
            },
            vector=vector,
        )
        self._refs[neo4j_id] = str(uuid)

    def _add_entity_relation(self, rel: Dict[str, Any]):
        """Entity -> Entity relatesTo reference (mirrors Neo4j RELATES_TO edge)."""
        src_key  = f"{rel.get('source')}||Unknown"
        tgt_key  = f"{rel.get('target')}||Unknown"
        src_uuid = self._refs.get(src_key)
        tgt_uuid = self._refs.get(tgt_key)
        if src_uuid and tgt_uuid:
            self.client.collections.get("Entity").data.reference_add(
                from_uuid=src_uuid, from_property="relatesTo", to=tgt_uuid,
            )

    # ------------------- Cross-reference wiring ------------------- #

    def _wire_references(self, chunks, filing_id: str):
        from app.data.processors.chunking import ChunkType

        doc_uuid = self._refs.get(filing_id)

        for chunk in chunks:
            chunk_uuid = self._refs.get(chunk.id)
            if not chunk_uuid:
                continue

            if chunk.type == ChunkType.SECTION and doc_uuid:
                self.client.collections.get("Section").data.reference_add(
                    from_uuid=chunk_uuid, from_property="partOfDocument", to=doc_uuid,
                )

            if chunk.type in (ChunkType.TEXT, ChunkType.TABLE):
                # partOfSection
                parent_uuid = self._refs.get(chunk.parent_id)
                if parent_uuid:
                    self.client.collections.get("Chunk").data.reference_add(
                        from_uuid=chunk_uuid, from_property="partOfSection", to=parent_uuid,
                    )
                # partOfDocument
                if doc_uuid:
                    self.client.collections.get("Chunk").data.reference_add(
                        from_uuid=chunk_uuid, from_property="partOfDocument", to=doc_uuid,
                    )
                # mentionsEntities
                for entity in chunk.entities:
                    ent_key  = f"{entity.get('name', 'Unknown')}||{entity.get('type', 'Unknown')}"
                    ent_uuid = self._refs.get(ent_key)
                    if ent_uuid:
                        self.client.collections.get("Chunk").data.reference_add(
                            from_uuid=chunk_uuid, from_property="mentionsEntities", to=ent_uuid,
                        )

        # Entity -> Document
        for neo4j_id, uuid in self._refs.items():
            if "||" in neo4j_id and doc_uuid:   # entity keys contain "||"
                self.client.collections.get("Entity").data.reference_add(
                    from_uuid=uuid, from_property="belongsToDocument", to=doc_uuid,
                )
