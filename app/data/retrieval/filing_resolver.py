from typing import Optional, List, Dict
from app.data.retrieval.graph_retriever import GraphRetriever


class FilingResolver:
    """Resolves company names to actual filing_ids stored in Neo4j."""

    # Static alias map for common names to ticker fragments
    ALIASES: Dict[str, List[str]] = {
        "google":    ["GOOG", "GOOGL"],
        "alphabet":  ["GOOG", "GOOGL"],
        "microsoft": ["MSFT"],
        "nvidia":    ["NVDA"],
        "oracle":    ["ORCL"],
        "meta":      ["META"],
    }

    def __init__(self):
        self._graph   = GraphRetriever()
        self._cache:  Dict[str, str] = {}
        self._all:    List[Dict]     = []
        self._refresh()

    def _refresh(self):
        """Load all filing_ids from Neo4j into cache."""
        self._all = self._graph.list_documents()
        self._cache = {}
        for doc in self._all:
            fid = doc.get("filing_id", "")
            company = (doc.get("company_name") or "").lower()
            # Map company_name to filing_id
            self._cache[company] = fid
            # Map filing_id variants to itself
            self._cache[fid.lower()] = fid
            self._cache[fid.lower().replace(".md", "")] = fid

    def resolve(self, name: str) -> Optional[str]:
        """Resolve a name/alias/filing_id to the exact filing_id in the DB."""
        self._refresh()
        key = name.strip().lower().replace("_", "").replace(".md", "").replace(" ", "")

        # Direct cache hit
        if key in self._cache:
            return self._cache[key]

        # Alias lookup
        for alias, tickers in self.ALIASES.items():
            if alias in key or key in alias:
                for ticker in tickers:
                    for fid in self._cache.values():
                        if ticker.upper() in fid.upper():
                            return fid

        # Fuzzy substring match against all filing_ids
        for doc in self._all:
            fid = doc.get("filing_id", "")
            if key in fid.lower().replace("_", "").replace(".md", ""):
                return fid

        return None

    def resolve_many(self, names: List[str]) -> List[str]:
        """Resolve a list of company names to filing_ids, skipping unresolved."""
        results = []
        for name in names:
            fid = self.resolve(name)
            if fid:
                results.append(fid)
        return results

    def list_all(self) -> List[Dict]:
        self._refresh()
        return self._all

    def close(self):
        self._graph.close()
