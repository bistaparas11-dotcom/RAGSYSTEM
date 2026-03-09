from openai import OpenAI
from app.config import settings


class BGEM3Embedder:
    """
    Wraps the NVIDIA serverless BGE-M3 endpoint.
    """
    MODEL = "baai/bge-m3"

    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=settings.NVIDIA_NIM_API,
            base_url="https://integrate.api.nvidia.com/v1",
        )

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.MODEL,
            encoding_format="float",
            extra_body={"truncate": "END"},   # truncate instead of error on long text
        )
        return response.data[0].embedding

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.MODEL,
            encoding_format="float",
            extra_body={"truncate": "END"},
        )
        return [d.embedding for d in response.data]