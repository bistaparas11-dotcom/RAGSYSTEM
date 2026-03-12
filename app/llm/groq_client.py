from openai import OpenAI
from typing import List, Dict, Any, Optional, Iterator

from groq import Groq
from app.config import settings


class GroqClient:
    MODEL = "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768", "gemma2-9b-it"

    SYSTEM_PROMPT = """You are an expert financial analyst specialized in SEC 10-K filings analysis.

You have access to retrieved context from a financial knowledge graph containing:
- SEC 10-K filings from major technology companies
- Extracted entities (executives, products, competitors, risks)
- Cross-company relationship graphs
- AI-generated summaries

Guidelines:
- Ground every claim in the provided context. If the context doesn't support an answer, say so clearly.
- When comparing companies, use specific data points from the filings.
- For financial figures, always cite the fiscal year and filing source.
- Highlight material risks when relevant.
- Be concise but thorough. Use bullet points for lists of facts.
"""

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def chat(
        self,
        messages,
        temperature=0.2,
        max_tokens=2048,
        stream=False
    ):
        full_messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + messages
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            return self._stream_tokens(response)
        return response.choices[0].message.content

    def complete(
        self,
        user_message: str,
        context: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Single-turn convenience wrapper with optional context injection."""
        content = user_message
        if context:
            content = f"Use the following context to answer the question.\n\n{context}\n\nQuestion: {user_message}"

        return self.chat(
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
        )

    @staticmethod
    def _stream_tokens(response) -> Iterator[str]:
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
