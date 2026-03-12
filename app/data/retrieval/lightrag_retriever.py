from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from app.data.retrieval.graph_retriever import GraphRetriever
from app.data.retrieval.weaviate_retriever import WeaviateRetriever


@dataclass
class RetrievedContext:
    """Unified context bundle passed to the LLM."""
    query: str
    chunks:          List[Dict[str, Any]] = field(default_factory=list)
    entities:        List[Dict[str, Any]] = field(default_factory=list)
    graph_context:   List[Dict[str, Any]] = field(default_factory=list)
    neighbour_chunks:List[Dict[str, Any]] = field(default_factory=list)
    source_filings:  List[str]            = field(default_factory=list)

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
            for e in self.entities:
                line = f"* [{e.get('entity_type', e.get('type', 'Unknown'))}] {e.get('name', '')}"
                if e.get("description"):
                    line += f": {e['description']}"
                parts.append(line)

        if self.graph_context:
            parts.append("\n=== GRAPH RELATIONSHIPS ===")
            for g in self.graph_context:
                parts.append(
                    f"* {g.get('name', '')} ({g.get('type', '')}) — {g.get('relationship', '')}"
                )

        if self.chunks:
            parts.append("\n=== MAIN CHUNKS ===")
            for i, c in enumerate(self.chunks, 1):
                filing = c.get("filing_id", "")
                section = c.get("section", "")
                parts.append(f"\n[{i}] Filing: {filing} | Section: {section}")
                parts.append(c.get("content") or c.get("summary") or "")

        if self.neighbour_chunks:
            parts.append("\n=== SURROUNDING CONTEXT ===")
            for c in self.neighbour_chunks:
                parts.append(c.get("content", ""))

        return "\n".join(parts)


class LightRAGRetriever:
    """
    LightRAG-style hybrid retriever:
      1. Dense vector search  (Weaviate)  → top-k chunks + entities
      2. Graph traversal      (Neo4j)     → entity context + neighbours
      3. Context fusion                   → deduped, ranked RetrievedContext

    Retrieval modes
    ---------------
    - "local"  : entity + chunk retrieval for a specific filing
    - "global" : cross-company entity & relationship traversal
    - "hybrid" : both (default, recommended)
    """

    def __init__(self):
        self.vector  = WeaviateRetriever()
        self.graph   = GraphRetriever()

    def close(self):
        self.vector.close()
        self.graph.close()

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",           # "local" | "global" | "hybrid"
        top_k: int = 8,
        filing_id: Optional[str] = None,
        expand_neighbours: bool = True,
    ) -> RetrievedContext:

        ctx = RetrievedContext(query=query)

        # Vector search
        vec_chunks   = self.vector.search_chunks(
            query, top_k=top_k, filing_id=filing_id
        )
        vec_entities = self.vector.search_entities(
            query, top_k=5, filing_id=filing_id
        )
        ctx.chunks   = vec_chunks
        ctx.entities = vec_entities

        # Graph traversal
        if mode in ("local", "hybrid"):
            ctx = self._local_graph_expansion(ctx, filing_id)

        if mode in ("global", "hybrid"):
            ctx = self._global_entity_expansion(ctx)

        # Neighbour window expansion
        if expand_neighbours and vec_chunks:
            seen_ids = {c.get("chunk_id") for c in vec_chunks}
            neighbour_chunks = []
            for c in vec_chunks[:3]:
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
        self, ctx: RetrievedContext, filing_id: Optional[str]
    ) -> RetrievedContext:
        """For each vector-retrieved entity, fetch its graph context."""
        graph_ctx = []
        seen_entities = set()

        for entity in ctx.entities[:5]:
            name = entity.get("name")
            if not name or name in seen_entities:
                continue
            seen_entities.add(name)

            entity_ctx = self.graph.get_entity_context(name)
            related = entity_ctx.get("related_entities", [])
            graph_ctx.extend(related)

            # Merge graph chunks into ctx.chunks
            existing_ids = {c.get("chunk_id") for c in ctx.chunks}
            for gc in entity_ctx.get("mentioning_chunks", []):
                if gc.get("chunk_id") not in existing_ids:
                    ctx.chunks.append(gc)
                    existing_ids.add(gc.get("chunk_id"))

        ctx.graph_context = graph_ctx
        return ctx

    def _global_entity_expansion(self, ctx: RetrievedContext) -> RetrievedContext:
        """For each entity, find all other filings that mention it."""
        cross_refs = []
        for entity in ctx.entities[:3]:
            name = entity.get("name")
            if not name:
                continue
            companies = self.graph.get_cross_company_entities(name)
            for c in companies:
                c["entity_name"] = name
                cross_refs.append(c)

        # Dedupe and append to graph_context
        seen = {g.get("name") for g in ctx.graph_context}
        for ref in cross_refs:
            key = f"{ref.get('entity_name')}::{ref.get('filing_id')}"
            if key not in seen:
                ctx.graph_context.append(ref)
                seen.add(key)

        return ctx