import requests
from typing import List, Union
from app.config import settings


class BGEM3Embedder:
    """Wraps the Modal-deployed BGE-M3 embedding endpoint."""
    BASE_URL = settings.BGE_M3_URL

    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self.session = requests.Session()

    def embed(self, text: str, normalize: bool = True, max_length: int = 8192) -> List[float]:
        """Embed a single text."""
        payload = {
            "input": [text],
            "normalize_embeddings": normalize,
            "max_length": max_length
        }
        response = self.session.post(
            f"{self.BASE_URL}/embed",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]

    def embed_many(
        self, 
        texts: List[str], 
        normalize: bool = True, 
        max_length: int = 8192,
        batch_size: int = 16  # Optional: split large lists to avoid timeout
    ) -> List[List[float]]:
        """Embed multiple text strings with optional batching."""
        all_embeddings = []
        
        # Process in batches to stay within timeout limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "input": batch,
                "normalize_embeddings": normalize,
                "max_length": max_length
            }
            response = self.session.post(
                f"{self.BASE_URL}/embed",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend(data["embeddings"])
        
        return all_embeddings

    def health_check(self) -> bool:
        """Check if the Modal service is healthy."""
        try:
            response = self.session.get(f"{self.BASE_URL}/health", timeout=30)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_model_info(self) -> dict:
        """Fetch model metadata from the service."""
        response = self.session.get(f"{self.BASE_URL}/model_info", timeout=30)
        response.raise_for_status()
        return response.json()