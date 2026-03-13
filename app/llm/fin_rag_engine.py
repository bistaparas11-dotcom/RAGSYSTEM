from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator

from app.llm.groq_client import GroqClient
from app.data.retrieval.filing_resolver import FilingResolver
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
        self.resolver = FilingResolver()
        self._history: List[Dict[str, str]] = []

    def close(self):
        self.retriever.close()
        self.resolver.close()

    def ask(
        self,
        question: str,
        mode: str = "hybrid",
        filing_id: Optional[str] = None,
        top_k: int = 10,
        rerank_top_k: int = 5,
        stream: bool = False,
    ) -> QAResponse | Iterator[str]:
        """Single question -> retriever -> llm answer.

        Args:
            question: Natural language question.
            mode: "local" | "global" | "hybrid"
            filing_id: Restrict retrieval to a single filing (optional).
            top_k: Number of chunks to retrieve.
            rerank_top_k: Number of chunks to rerank.
            stream: Stream tokens back instead of waiting for full answer.

        Returns:
            QAResponse (or token iterator if stream=True).
        """
        # ------------------- 0. Resolve filing ID if needed ------------------- #
        resolved_fid = None
        if filing_id:
            resolved_fid = self.resolver.resolve(filing_id)
            if not resolved_fid:
                print(f" [!] Could not resolve filing_id: {filing_id}")
        
        if not resolved_fid:
            resolved_fid = self._detect_filing_from_question(question)

        # ------------------- 1. Retrieve ------------------- #
        ctx = self.retriever.retrieve(
            query=question,
            mode=mode,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            filing_id=resolved_fid,
        )

        # ------------------- 2. Build prompt ------------------- #
        context_text = ctx.to_prompt_text()

        if not ctx.chunks and not ctx.entities:
            context_text = (
                "No relevant passages were found in the knowledge graph "
                "for the available filings"
            )

        user_message = (
            f"Context from SEC 10-K financial filings:\n\n{context_text}\n\n"
            f"Question: {question}"
        )

        # ------------------- 3. Append to conversation history ------------------- #
        trimmed_history = self._history[-4:] if len(self._history) > 4 else self._history
        messages = trimmed_history + [{"role": "user", "content": user_message}]

        # ------------------- 4. Call Groq ------------------- #
        if stream:
            return self.llm.chat(messages, stream=True)

        answer = self.llm.chat(messages)
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": answer})

        return QAResponse(
            question=question,
            answer=answer,
            context=ctx,
            source_filings=ctx.source_filings,
            retrieval_mode=mode,
        )
    
    def _detect_filing_from_question(self, question: str) -> Optional[str]:
        """Try to resolve a single company name mentioned in the question."""
        q_lower = question.lower()

        for alias in self.resolver.ALIASES:
            if alias in q_lower:
                resolved = self.resolver.resolve(alias)
                if resolved:
                    return resolved
        
        # Check actual company name from DB
        for doc in self.resolver.list_all():
            company = (doc.get("company_name") or "").lower()
            if company and company in q_lower:
                return doc.get("filing_id")

        return None

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
        return self.ask(question, mode="global", top_k=12, rerank_top_k=6)

    def summarise_risks(self, filing_id: str) -> QAResponse:
        """Return a structured risk summary for a single filing."""
        question = (
            "Summarise the top material risks disclosed in this filing. "
            "Group them by category (financial, operational, regulatory, competitive). "
            "Include specific language from the Risk Factors section."
        )
        return self.ask(question, mode="local", filing_id=filing_id, top_k=10, rerank_top_k=5)

    def extract_financials(self, filing_id: str) -> QAResponse:
        """Extract key financial metrics from a filing."""
        question = (
            "Extract and present the key financial metrics from this filing: "
            "revenue, net income, operating margin, YoY growth, cash position, "
            "and any forward guidance. Present as a structured summary."
        )
        return self.ask(question, mode="local", filing_id=filing_id, top_k=10, rerank_top_k=5)
