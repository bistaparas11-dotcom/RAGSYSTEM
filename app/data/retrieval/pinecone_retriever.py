from typing import List, Dict, Any, Optional
from pinecone import Pinecone

from app.config import settings
from app.utils.embedding_utils import BGEM3Embedder


class PineconeRetriever:
    """Vector similarity search over Pinecone index."""

    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
        self.embedder = BGEM3Embedder()

    def close(self):
        # Pinecone client doesn't explicitly need closing like Weaviate's gRPC client
        pass

    def search_chunks(
        self,
        query: str,
        top_k: int = 10,
        filing_id: Optional[str] = None,
        section: Optional[str] = None,
        is_financial: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search over Chunk records."""
        vector = self.embedder.embed(query)
        
        filter_dict = {"type": "Chunk"}
        if filing_id:
            filter_dict["filing_id"] = filing_id
        if section:
            filter_dict["section"] = section
        if is_financial is not None:
            filter_dict["is_financial"] = is_financial

        results = self.index.query(
            vector=vector,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )

        return [self._hit_to_dict(hit) for hit in results.matches]

    def search_sections(
        self,
        query: str,
        top_k: int = 5,
        filing_id: Optional[str] = None,
        is_risk: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Dense search over Section records."""
        vector = self.embedder.embed(query)
        
        filter_dict = {"type": "Section"}
        if filing_id:
            filter_dict["filing_id"] = filing_id
        if is_risk is not None:
            filter_dict["is_risk"] = is_risk

        results = self.index.query(
            vector=vector,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        return [self._hit_to_dict(hit) for hit in results.matches]

    def search_entities(
        self,
        query: str,
        top_k: int = 5,
        entity_type: Optional[str] = None,
        filing_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Dense search over Entity records."""
        vector = self.embedder.embed(query)
        
        filter_dict = {"type": "Entity"}
        if entity_type:
            filter_dict["entity_type"] = entity_type
        if filing_id:
            filter_dict["filing_id"] = filing_id

        results = self.index.query(
            vector=vector,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        return [self._hit_to_dict(hit) for hit in results.matches]

    def _hit_to_dict(self, hit) -> Dict[str, Any]:
        """Convert Pinecone hit to a dictionary format compatible with the app."""
        result = dict(hit.metadata)
        result["_distance"] = hit.score # Pinecone uses score (similarity), Weaviate uses distance
        result["_id"] = hit.id
        return result
