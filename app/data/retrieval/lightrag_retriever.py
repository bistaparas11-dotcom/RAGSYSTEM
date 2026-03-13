from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from app.config import settings
from app.data.retrieval.reranker import Reranker
from app.data.retrieval.graph_retriever import GraphRetriever
from app.data.retrieval.weaviate_retriever import WeaviateRetriever


@dataclass
class RetrievedContext:
    """Unified context bundle passed to the LLM."""
    query:            str
    chunks:           List[Dict[str, Any]] = field(default_factory=list)
    entities:         List[Dict[str, Any]] = field(default_factory=list)
    graph_context:    List[Dict[str, Any]] = field(default_factory=list)
    neighbour_chunks: List[Dict[str, Any]] = field(default_factory=list)
    source_filings:   List[str]            = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Serialize all context into a clean prompt block for the LLM.
        
        Returns:
            str: === RELEVANT ENTITIES ===
                 * [ENTITY_TYPE] ENTITY_NAME: DESCRIPTION
                 
                 === GRAPH RELATIONSHIPS ===
                 * ENTITY_NAME (TYPE) — RELATIONSHIP
                 
                 === MAIN CHUNKS ===
                 [1] Filing: FILING_ID | Section: SECTION
                 CONTENT
                 
                 === SURROUNDING CONTEXT ===
                 CONTENT
        """
        parts = []

        if self.entities:
            parts.append("=== RELEVANT ENTITIES ===")
            for e in self.entities[:5]:
                line = f"* [{e.get('entity_type', e.get('type', 'Unknown'))}] {e.get('name', '')}"
                if e.get("description"):
                    line += f": {e['description'][:200]}"
                parts.append(line)

        if self.graph_context:
            parts.append("\n=== GRAPH RELATIONSHIPS ===")
            for g in self.graph_context[:8]:
                parts.append(
                    f"* {g.get('name', '')} ({g.get('type', '')}) — {g.get('relationship', '')}"
                )

        if self.chunks:
            parts.append("\n=== MAIN CHUNKS ===")
            for i, c in enumerate(self.chunks, 1):
                filing = c.get("filing_id", "")
                section = c.get("section", "")
                score = c.get("_rerank_score")
                score_str = f" (Score: {score:.2f})" if score is not None else ""
                content = (c.get("content") or c.get("summary") or "")
                parts.append(f"\n[{i}] Filing: {filing} | Section: {section}{score_str}")
                parts.append(content)

        if self.neighbour_chunks:
            parts.append("\n=== SURROUNDING CONTEXT ===")
            for c in self.neighbour_chunks[:3]:
                parts.append(c.get("content", ""))

        return "\n".join(parts)


class LightRAGRetriever:
    """
    LightRAG-style hybrid retriever:
      1. Dense vector search  (Weaviate)  -> top-k chunks + entities
      2. Graph traversal      (Neo4j)     -> entity context + neighbours
      3. Rerank               (Reranker)  -> keep only top relevant chunks
    """

    def __init__(self):
        self.vector  = WeaviateRetriever()
        self.graph   = GraphRetriever()
        self.reranker = Reranker(use_cross_encoder=True)

    def close(self):
        self.vector.close()
        self.graph.close()

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",   # "local" | "global" | "hybrid"
        top_k: int = 10,
        rerank_top_k: int = 5,
        filing_id: Optional[str] = None,
        expand_neighbours: bool = True,
        min_rerank_score: float = 0.25,
    ) -> RetrievedContext:

        ctx = RetrievedContext(query=query)

        # Vector search
        vec_chunks   = self.vector.search_chunks(
            query, top_k=top_k, filing_id=filing_id
        )
        vec_entities = self.vector.search_entities(
            query, top_k=5, filing_id=filing_id
        )

        # Graph traversal
        if mode in ("local", "hybrid"):
            vec_chunks, vec_entities = self._local_graph_expansion(
                query, vec_chunks, vec_entities, filing_id
            )

        if mode in ("global", "hybrid"):
            vec_entities = self._global_entity_expansion(vec_entities)
        
        reranked = self.reranker.rerank(
            query=query,
            chunks=vec_chunks,
            top_k=rerank_top_k,
            min_score=min_rerank_score,
        )

        ctx.chunks = reranked
        ctx.entities = vec_entities

        # Neighbour window expansion
        if expand_neighbours and reranked:
            seen_ids = {c.get("chunk_id") for c in reranked}
            neighbour_chunks = []
            for c in reranked[:2]:
                cid = c.get("chunk_id")
                if cid:
                    neighbours = self.graph.get_chunk_with_neighbours(cid, window=1)
                    for n in neighbours:
                        if n.get("chunk_id") not in seen_ids:
                            neighbour_chunks.append(n)
                            seen_ids.add(n.get("chunk_id"))
            ctx.neighbour_chunks = neighbour_chunks

        # Collect unique source filings
        all_chunks = ctx.chunks + ctx.neighbour_chunks
        ctx.source_filings = list({
            c.get("filing_id") for c in all_chunks if c.get("filing_id")
        })

        return ctx

    # ------------------- Graph expansion helper functions ------------------- #

    def _local_graph_expansion(
        self, query, chunks, entities, filing_id
    ):
        """For each vector-retrieved entity, fetch its graph context."""
        seen_chunk_ids = {c.get("chunk_id") for c in chunks}
        seen_entity_names = set()
        extra_chunks = []

        for entity in entities[:4]:
            name = entity.get("name")
            if not name or name in seen_entity_names:
                continue
            seen_entity_names.add(name)

            entity_ctx = self.graph.get_entity_context(name)
            for gc in entity_ctx.get("mentioning_chunks", []):
                if filing_id and gc.get("filing_id") == filing_id:
                    continue
                if gc.get("chunks_id") not in seen_chunk_ids:
                    extra_chunks.append(gc)
                    seen_chunk_ids.add(gc.get("chunk_id"))

        return chunks + extra_chunks, entities

    def _global_entity_expansion(self, entities):
        """For each entity, find all other filings that mention it."""
        seen = {e.get("name") for e in entities}
        extra = []

        for entity in entities[:3]:
            name = entity.get("name")
            if not name:
                continue
            companies = self.graph.get_cross_company_entities(name)
            for c in companies:
                key = f"{name}::{c.get('filing_id')}"
                if key not in seen:
                    extra.append({
                        "name": name,
                        "type": "cross_filing",
                        "relationship": f"Mentioned in {c.get('company_name')} {c.get('fiscal_year')}",
                        "filing_id": c.get("filing_id"),
                    })
                seen.add(key)

        return entities + extra
