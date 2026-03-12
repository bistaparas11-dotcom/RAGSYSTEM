from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator

from app.llm.groq_client import GroqClient
from app.data.retrieval.lightrag_retriever import LightRAGRetriever, RetrievedContext


@dataclass
class QAResponse:
    question: str
    answer: str
    context: RetrievedContext
    source_filings: List[str] = field(default_factory=list)
    retrieval_mode: str = "hybrid"


class FinRAGEngine:
    """End-to-end RAG pipeline, that maintains conversation history."""

    def __init__(self):
        self.retriever = LightRAGRetriever()
        self.llm = GroqClient()
        self._history: List[Dict[str, str]] = []

    def close(self):
        self.retriever.close()

    def ask(
        self,
        question: str,
        mode: str = "hybrid",
        filing_id: Optional[str] = None,
        top_k: int = 8,
        stream: bool = False,
    ) -> QAResponse | Iterator[str]:
        """Single question -> retriever -> llm answer.

        Args:
            question: Natural language question.
            mode: "local" | "global" | "hybrid"
            filing_id: Restrict retrieval to a single filing (optional).
            top_k: Number of chunks to retrieve.
            stream: Stream tokens back instead of waiting for full answer.

        Returns:
            QAResponse (or token iterator if stream=True).
        """
        # ------------------- 1. Retrieve ------------------- #
        ctx = self.retriever.retrieve(
            query=question,
            mode=mode,
            top_k=top_k,
            filing_id=filing_id,
        )

        # ------------------- 2. Build prompt ------------------- #
        context_text = ctx.to_prompt_text()
        user_message = (
            f"Context from financial filings:\n\n{context_text}\n\n"
            f"Question: {question}"
        )

        # ------------------- 3. Append to conversation history ------------------- #
        self._history.append({"role": "user", "content": user_message})

        # ------------------- 4. Call Groq ------------------- #
        if stream:
            return self.llm.chat(self._history, stream=True)

        answer = self.llm.chat(self._history)
        self._history.append({"role": "assistant", "content": answer})

        return QAResponse(
            question=question,
            answer=answer,
            context=ctx,
            source_filings=ctx.source_filings,
            retrieval_mode=mode,
        )

    def reset_history(self):
        """Clear conversation memory for a new session."""
        self._history = []

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._history)

    # -------------------- Comparison query ------------------- #

    def compare_companies(
        self,
        companies: List[str],
        topic: str,
    ) -> QAResponse:
        """Cross-company comparison using global retrieval mode."""
        question = (
            f"Compare {', '.join(companies)} on the topic of '{topic}'. "
            f"Use specific figures and dates from their 10-K filings. "
            f"Highlight key differences and similarities."
        )
        return self.ask(question, mode="global", top_k=12)

    def summarise_risks(self, filing_id: str) -> QAResponse:
        """Return a structured risk summary for a single filing."""
        question = (
            "Summarise the top material risks disclosed in this filing. "
            "Group them by category (financial, operational, regulatory, competitive). "
            "Include specific language from the Risk Factors section."
        )
        return self.ask(question, mode="local", filing_id=filing_id, top_k=10)

    def extract_financials(self, filing_id: str) -> QAResponse:
        """Extract key financial metrics from a filing."""
        question = (
            "Extract and present the key financial metrics from this filing: "
            "revenue, net income, operating margin, YoY growth, cash position, "
            "and any forward guidance. Present as a structured summary."
        )
        return self.ask(question, mode="local", filing_id=filing_id, top_k=10)
