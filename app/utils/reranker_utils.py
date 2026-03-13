import requests
from typing import List, Dict, Any
from app.config import settings


class NimReranker:
    """Nvidia NIM Reranker."""
    
    MODEL = "nv-rerank-qa-mistral-4b:1"
    INVOKE_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"

    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {settings.NVIDIA_NIM_API}",
            "Accept": "application/json",
        }

    def rerank_run(self, query: str, passages: List[str]) -> List[Dict[str, Any]]:
        """
        Rerank a list of passages for a given query using Nvidia NIM.
        
        Args:
            query: The question or query string.
            passages: A list of chunk strings to rerank.
            
        Returns:
            A list of dictionaries containing the text, the ranking score (logit), 
            and the original index, sorted by score in descending order.
        """
        if not passages:
            return []

        payload = {
            "model": self.MODEL,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages]
        }

        response = self.session.post(self.INVOKE_URL, headers=self.headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        rankings = data.get("rankings", [])
        
        results = []
        for item in rankings:
            idx = item["index"]
            results.append({
                "text": passages[idx],
                "score": item["logit"],
                "index": idx
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
            
        return results