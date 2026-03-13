import math
from typing import List, Dict, Any, Tuple


class Reranker:
    """Two-stage reranker for retrieved chunks.

    Stage 1 - keyword overlap score (no dependency, fast)
    Stage 2 - cross-encoder via NVIDIA NIM (optional, higher quality)

    Falls back to stage 1 if cross-encoder unavailable.
    """

    def __init__(self, use_cross_encoder: bool = False):
        self.use_cross_encoder = use_cross_encoder
        self._ce_client = None

        if self.use_cross_encoder:
            try:
                from app.utils.reranker_utils import NimReranker
                self._ce_client = NimReranker()
            except Exception:
                self.use_cross_encoder = False

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Score and filter chunks, return top_k most relevant.

        Args:
            query:     The user question.
            chunks:    Retrieved chunk dicts (must have 'content' or 'summary').
            top_k:     How many to keep after reranking.
            min_score: Drop chunks below this relevance score (0.0–1.0).

        Returns:
            Sorted list of chunk dicts with added '_rerank_score' field.
        """
        if not chunks:
            return []

        if self.use_cross_encoder:
            scored = self._cross_encoder_score(query, chunks)
        else:
            scored = self._keyword_score(query, chunks)

        # Filter low-quality chunks
        scored = [(score, chunk) for score, chunk in scored if score >= min_score]

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored[:top_k]:
            chunk["_rerank_score"] = round(score, 4)
            results.append(chunk)

        return results

    # ---------- Stage 1: Simplified BM25 scoring ---------- #
    @staticmethod
    def _keyword_score(
        query: str, chunks: List[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Simple TF-IDF keyword overlap score.
        Scores based on:
          - query term coverage in chunk text
          - boost for financial keywords when query is financial
          - boost for summary match vs raw content
        """
        query_terms = set(query.lower().split())

        # Remove stopwords
        stopwords = {
            "a","an","the","is","are","was","were","in","of","to","and",
            "or","for","with","on","at","by","from","that","this","what",
            "how","why","when","which","who","tell","me","about","give",
            "show","list","explain",
        }
        query_terms -= stopwords

        financial_terms = {
            "revenue","income","profit","loss","margin","cash","debt",
            "equity","growth","guidance","segment","quarterly","annual",
            "fiscal","earnings","ebitda","capex","dividend","buyback",
        }
        risk_terms = {
            "risk","uncertainty","litigation","regulation","compliance",
            "cybersecurity","competition","macroeconomic","inflation",
        }

        scored = []
        for chunk in chunks:
            text = (
                (chunk.get("summary") or "") + " " +
                (chunk.get("content") or "")
            ).lower()

            words = set(text.split())

            # Base: fraction of query terms found in chunk
            if not query_terms:
                base_score = 0.5
            else:
                hits = sum(1 for t in query_terms if t in text)
                base_score = hits / len(query_terms)

            # Boost: financial/risk terms in both query and chunk
            q_financial = bool(query_terms & financial_terms)
            q_risk = bool(query_terms & risk_terms)
            c_financial = bool(words & financial_terms)
            c_risk = bool(words & risk_terms)

            boost = 0.0
            if q_financial and c_financial:
                boost += 0.2
            if q_risk and c_risk:
                boost += 0.2

            # Penalise very short chunks
            text_len = len(text.split())
            if text_len < 30:
                boost -= 0.3

            # Vector distance bonus
            dist = chunk.get("_distance")
            if dist is not None:
                # distance 0=perfect, 2=worst; convert to 0–0.3 bonus
                dist_bonus = max(0.0, 0.3 * (1.0 - dist / 2.0))
                boost += dist_bonus

            score = min(1.0, max(0.0, base_score + boost))
            scored.append((score, chunk))

        return scored

    # ---------- Stage 2: NVIDIA NIM cross-encoder ---------- #
    def _cross_encoder_score(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Rerank chunks using Nvidia NIM reranker"""
        try:
            passages = [
                (chunk.get("summary") or chunk.get("content") or "")
                for chunk in chunks
            ]
            results = self._ce_client.rerank_run(query, passages)
            
            # results is already sorted by score (logits)
            scores = [res["score"] for res in results]
            ranked_chunks = [chunks[res["index"]] for res in results]
            
            return list(zip(scores, ranked_chunks))

        except Exception as e:
            print(f"[Reranker] failed ({e}), falling back to keyword scoring.")
            return self._keyword_score(query, chunks)