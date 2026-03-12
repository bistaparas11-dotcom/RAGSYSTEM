import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from typing import List, Dict, Any, Optional

from app.config import settings
from app.utils.embedding_utils import BGEM3Embedder


class WeaviateRetriever:
    """Vector similarity search over Weaviate collections."""

    def __init__(self):
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.WEAVIATE_URL,
            auth_credentials=Auth.api_key(settings.WEAVIATE_API_KEY),
        )
        self.embedder = BGEM3Embedder(settings.NVIDIA_NIM_API)

    def close(self):
        self.client.close()

    def search_chunks(
        self,
        query: str,
        top_k: int = 10,
        filing_id: Optional[str] = None,
        section: Optional[str] = None,
        is_financial: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search over Chunk collection.

        Returns:
            list[dict]: A ranked list of chunk dicts with score.
        """
        vector = self.embedder.embed(query)
        col = self.client.collections.get("Chunk")

        filters = self._build_filters(
            filing_id=filing_id,
            section=section,
            is_financial=is_financial,
        )

        results = col.query.near_vector(
            near_vector=vector,
            limit=top_k,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )

        return [self._obj_to_dict(o) for o in results.objects]

    def search_sections(
        self,
        query: str,
        top_k: int = 5,
        filing_id: Optional[str] = None,
        is_risk: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Dense search over Section collection.
        
        Returns:
            list[dict]: A ranked list of section dicts with score.    
        """
        vector = self.embedder.embed(query)
        col = self.client.collections.get("Section")

        filters = self._build_filters(filing_id=filing_id, is_risk=is_risk)

        results = col.query.near_vector(
            near_vector=vector,
            limit=top_k,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )
        return [self._obj_to_dict(o) for o in results.objects]

    def search_entities(
        self,
        query: str,
        top_k: int = 5,
        entity_type: Optional[str] = None,
        filing_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Dense search over Entity collection.
        
        Returns:
            list[dict]: A ranked list of entity dicts with score.
        """
        vector = self.embedder.embed(query)
        col = self.client.collections.get("Entity")

        filters = self._build_filters(entity_type=entity_type, filing_id=filing_id)

        results = col.query.near_vector(
            near_vector=vector,
            limit=top_k,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )
        return [self._obj_to_dict(o) for o in results.objects]

    @staticmethod
    def _build_filters(**kwargs) -> Optional[Filter]:
        """Build a combined AND filter."""
        conditions = []
        for key, value in kwargs.items():
            if value is None:
                continue
            conditions.append(Filter.by_property(key).equal(value))

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return Filter.all_of(conditions)

    @staticmethod
    def _obj_to_dict(obj) -> Dict[str, Any]:
        result = dict(obj.properties)
        result["_distance"] = obj.metadata.distance if obj.metadata else None
        result["_uuid"] = str(obj.uuid)
        return result