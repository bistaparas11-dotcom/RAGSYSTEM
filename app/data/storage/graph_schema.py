import json
from neo4j import GraphDatabase
from typing import List, Dict, Any

from app.data.processors.chunking import Chunk, ChunkType


class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def build_graph(self, chunks: List[Chunk], filing_metadata: Dict[str, Any]):
        """
        Build a knowledge graph from a list of Chunks produced by SECChunker.

        Each Chunk already carries:
          - chunk.entities      → list of {name, type, description}
          - chunk.relationships → list of {source, relationship, target, descri ption}
          - chunk.summary       → 3-sentence summary string

        All edges are :RELATES_TO with a `type` property for flexibility.
        """
        with self.driver.session() as session:
            # 1. Document root
            doc_chunk = next((c for c in chunks if c.type == ChunkType.DOCUMENT), None)
            if doc_chunk:
                session.execute_write(self._create_document_node_tx, doc_chunk, filing_metadata)

            # 2. Section nodes
            for chunk in chunks:
                if chunk.type == ChunkType.SECTION:
                    session.execute_write(self._create_section_node_tx, chunk, filing_metadata["filing_id"])

            # 3. Text / table leaf chunk nodes
            for chunk in chunks:
                if chunk.type in (ChunkType.TEXT, ChunkType.TABLE):
                    session.execute_write(self._create_chunk_node_tx, chunk, filing_metadata["filing_id"])

                    for entity in chunk.entities:
                        entity_dict = {"name": entity, "type": "Unknown", "description": ""} if isinstance(entity, str) else entity
                        session.execute_write(self._create_entity_node_tx, entity_dict, filing_metadata["filing_id"])
                        session.execute_write(self._link_chunk_to_entity_tx, chunk.id, entity_dict.get("name", entity))

                    for rel in chunk.relationships:
                        session.execute_write(self._create_relationship_tx, rel)

    # --------------------- Document Node --------------------- #

    @staticmethod
    def _create_document_node_tx(tx, chunk: Chunk, metadata: Dict[str, Any]):
        """
        Creates a single :Document root node for the filing.
        chunk.content holds the pre-built description string from SECChunker.
        """
        query = """
        MERGE (d:Document {filing_id: $filing_id})
        SET d.company_name  = $company_name,
            d.filing_type   = $filing_type,
            d.fiscal_year   = $fiscal_year,
            d.content       = $content,
            d.is_root       = true,
            d.created_at    = datetime()
        """
        tx.run(
            query,
            filing_id    = metadata["filing_id"],
            company_name = metadata.get("company_name"),
            filing_type  = metadata.get("filing_type"),
            fiscal_year  = metadata.get("fiscal_year"),
            content      = chunk.content,
        )

    # --------------------- Section Node --------------------- #

    @staticmethod
    def _create_section_node_tx(tx, chunk: Chunk, filing_id: str):
        """
        Creates a :Section node and links it to its parent :Document.
        """
        query = """
        MERGE (s:Section {chunk_id: $chunk_id})
        SET s.section     = $section,
            s.filing_id   = $filing_id,
            s.content     = $content,
            s.char_count  = $char_count,
            s.has_tables  = $has_tables,
            s.is_risk     = $is_risk,
            s.updated_at  = datetime()

        WITH s
        MATCH (d:Document {filing_id: $filing_id})
        MERGE (s)-[:RELATES_TO {type: 'PART_OF_DOCUMENT'}]->(d)
        """
        tx.run(
            query,
            chunk_id   = chunk.id,
            section    = chunk.section,
            filing_id  = filing_id,
            content    = chunk.content,
            char_count = chunk.metadata.get("char_count", 0),
            has_tables = chunk.metadata.get("has_tables", False),
            is_risk    = chunk.metadata.get("is_risk", False),
        )

    # --------------------- Leaf Chunk Node --------------------- #

    @staticmethod
    def _create_chunk_node_tx(tx, chunk: Chunk, filing_id: str):
        """
        Creates a :Chunk node for a text or table leaf chunk and links it
        to its parent :Section (via parent_id) and to the :Document.
        """
        query = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.type          = $type,
            c.section       = $section,
            c.filing_id     = $filing_id,
            c.content       = $content,
            c.summary       = $summary,
            c.chunk_index   = $chunk_index,
            c.updated_at    = datetime()

        // Link to parent section
        WITH c
        MATCH (s:Section {chunk_id: $parent_id})
        MERGE (c)-[:RELATES_TO {type: 'PART_OF_SECTION'}]->(s)

        // Link to document
        WITH c
        MATCH (d:Document {filing_id: $filing_id})
        MERGE (c)-[:RELATES_TO {type: 'PART_OF_DOCUMENT'}]->(d)
        """
        tx.run(
            query,
            chunk_id    = chunk.id,
            type        = chunk.type.value,
            section     = chunk.section,
            filing_id   = filing_id,
            content     = chunk.content[:2000],
            summary     = chunk.summary,
            parent_id   = chunk.parent_id,
            chunk_index = chunk.metadata.get("chunk_index", 0),
        )

    # --------------------- Entity Node --------------------- #

    @staticmethod
    def _create_entity_node_tx(tx, entity: Dict[str, Any], filing_id: str):
        """
        Upserts an :Entity node from extract_sec_data output
        Links the entity back to its :Document so it is always reachable
        from the filing root.
        """
        if isinstance(entity, str):
            entity = {"name": entity, "type": "Unknown", "description": ""}

        query = """
        MERGE (e:Entity {name: $name, type: $type})
        SET e.description = $description,
            e.filing_id   = $filing_id,
            e.updated_at  = datetime()

        WITH e
        MATCH (d:Document {filing_id: $filing_id})
        MERGE (e)-[:RELATES_TO {type: 'BELONGS_TO_DOCUMENT'}]->(d)
        """
        tx.run(
            query,
            name        = entity.get("name", "Unknown"),
            type        = entity.get("type", "Unknown"),
            description = entity.get("description", ""),
            filing_id   = filing_id,
        )

    # --------------------- Chunk -> Entity edge --------------------- #

    @staticmethod
    def _link_chunk_to_entity_tx(tx, chunk_id: str, entity_name: str):
        """Creates a MENTIONS edge from a :Chunk to an :Entity."""
        query = """
        MATCH (c:Chunk   {chunk_id: $chunk_id})
        MATCH (e:Entity  {name: $entity_name})
        MERGE (c)-[:RELATES_TO {type: 'MENTIONS'}]->(e)
        """
        tx.run(query, chunk_id=chunk_id, entity_name=entity_name)

    # --------------------- Entity to Entity Relationship Edge --------------------- #

    @staticmethod
    def _create_relationship_tx(tx, rel: Dict[str, Any]):
        """
        Creates a :RELATES_TO edge between two :Entity nodes.

        extract_sec_data returns:
          {source, relationship, target, description}

        'relationship' maps to the edge `type` property.
        'description'  maps to the edge `context` property.
        """
        query = """
        MATCH (source:Entity {name: $source_name})
        MATCH (target:Entity {name: $target_name})
        MERGE (source)-[r:RELATES_TO {type: $rel_type}]->(target)
        SET r.context    = $context,
            r.updated_at = datetime()
        """
        tx.run(
            query,
            source_name = rel.get("source"),
            target_name = rel.get("target"),
            rel_type    = rel.get("relationship", "RELATED_TO"),
            context     = rel.get("description", "")[:500],
        )
