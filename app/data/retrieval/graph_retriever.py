from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional

from app.config import settings


class GraphRetriever:
    """Structured graph traversal over Neo4j."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )

    def close(self):
        self.driver.close()

    # ------------------- Entity-centric retrieval------------------- #

    def get_entity_context(
        self,
        entity_name: str,
        max_chunks: int = 5,
    ) -> Dict[str, Any]:
        """Get entity context based on entity name.

        Returns:
            dict: related entities and chunks that mention it
        """
        with self.driver.session() as session:
            entity = session.execute_read(self._fetch_entity_tx, entity_name)
            chunks = session.execute_read(self._fetch_entity_chunks_tx, entity_name, max_chunks)
            related = session.execute_read(self._fetch_related_entities_tx, entity_name)

        return {
            "entity": entity,
            "mentioning_chunks": chunks,
            "related_entities":  related,
        }

    def get_cross_company_entities(self, entity_name: str) -> List[Dict[str, Any]]:
        """Find all companies that mention a given entity."""

        query = """
        MATCH (e:Entity {name: $name})-[:RELATES_TO]->(d:Document)
        RETURN d.company_name AS company, d.filing_id AS filing_id,
               d.fiscal_year AS fiscal_year
        ORDER BY d.fiscal_year DESC
        """
        with self.driver.session() as session:
            result = session.run(query, name=entity_name)
            return [dict(r) for r in result]

    # ------------------- Chunk-centric retrieval------------------- #

    def get_chunk_with_neighbours(
        self,
        chunk_id: str,
        window: int = 1,
    ) -> List[Dict[str, Any]]:
        """Fetch a chunk plus its adjacent siblings within the same section."""

        query = """
        MATCH (target:Chunk {chunk_id: $chunk_id})
        MATCH (sibling:Chunk)
        WHERE sibling.filing_id = target.filing_id
          AND sibling.section = target.section
          AND abs(sibling.chunk_index - target.chunk_index) <= $window
        RETURN sibling.chunk_id AS chunk_id,
               sibling.content AS content,
               sibling.summary AS summary,
               sibling.chunk_index AS chunk_index,
               sibling.section AS section
        ORDER BY sibling.chunk_index
        """
        with self.driver.session() as session:
            result = session.run(query, chunk_id=chunk_id, window=window)
            return [dict(r) for r in result]

    def get_section_chunks(
        self,
        filing_id: str,
        section: str,
    ) -> List[Dict[str, Any]]:
        """Return all chunks from a specific section of a filing."""

        query = """
        MATCH (c:Chunk {filing_id: $filing_id, section: $section})
        RETURN c.chunk_id AS chunk_id, c.content AS content,
               c.summary AS summary, c.chunk_index AS chunk_index,
               c.type AS type
        ORDER BY c.chunk_index
        """
        with self.driver.session() as session:
            result = session.run(query, filing_id=filing_id, section=section)
            return [dict(r) for r in result]

    # ------------------- Document-level retrieval------------------- #

    def get_document_summary(self, filing_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document node metadata of a filing."""

        query = """
        MATCH (d:Document {filing_id: $filing_id})
        RETURN d.filing_id AS filing_id,
               d.company_name AS company_name,
               d.filing_type AS filing_type,
               d.fiscal_year AS fiscal_year,
               d.content AS content
        """
        with self.driver.session() as session:
            result = session.run(query, filing_id=filing_id)
            record = result.single()
            return dict(record) if record else None

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested filings."""
        query = """
        MATCH (d:Document)
        RETURN d.filing_id AS filing_id,
               d.company_name AS company_name,
               d.filing_type AS filing_type,
               d.fiscal_year AS fiscal_year
        ORDER BY d.company_name, d.fiscal_year DESC
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(r) for r in result]

    # ------------------- Graph path retrieval------------------- #

    def find_entity_paths(
        self,
        source_entity: str,
        target_entity: str,
        max_hops: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find shortest relationship paths between two entities."""

        query = """
        MATCH path = shortestPath(
            (a:Entity {name: $source})-[:RELATES_TO*1..$hops]-(b:Entity {name: $target})
        )
        RETURN [node in nodes(path) | coalesce(node.name, node.filing_id)] AS node_names,
               [rel  in relationships(path) | rel.type] AS rel_types,
               length(path) AS hops
        LIMIT 5
        """
        with self.driver.session() as session:
            result = session.run(query, source=source_entity, target=target_entity, hops=max_hops)
            return [dict(r) for r in result]

    def get_company_entities(
        self,
        filing_id: str,
        entity_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return all entities extracted from a filing."""

        query = """
        MATCH (e:Entity {filing_id: $filing_id})
        WHERE $entity_type IS NULL OR e.type = $entity_type
        RETURN e.name AS name, e.type AS type, e.description AS description
        ORDER BY e.type, e.name
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(
                query,
                filing_id=filing_id,
                entity_type=entity_type,
                limit=limit,
            )
            return [dict(r) for r in result]

    # ------------------- Helper functions ------------------- #

    @staticmethod
    def _fetch_entity_tx(tx, name: str):
        result = tx.run(
            "MATCH (e:Entity {name: $name}) RETURN e LIMIT 1", name=name
        )
        record = result.single()
        return dict(record["e"]) if record else None

    @staticmethod
    def _fetch_entity_chunks_tx(tx, name: str, limit: int):
        result = tx.run(
            """
            MATCH (c:Chunk)-[:RELATES_TO {type: 'MENTIONS'}]->(e:Entity {name: $name})
            RETURN c.chunk_id AS chunk_id, c.content AS content,
                   c.summary  AS summary,  c.section  AS section,
                   c.filing_id AS filing_id
            LIMIT $limit
            """,
            name=name, limit=limit,
        )
        return [dict(r) for r in result]

    @staticmethod
    def _fetch_related_entities_tx(tx, name: str):
        result = tx.run(
            """
            MATCH (e:Entity {name: $name})-[r:RELATES_TO]-(other:Entity)
            RETURN other.name AS name, other.type AS type,
                   r.type     AS relationship
            LIMIT 20
            """,
            name=name,
        )
        return [dict(r) for r in result]