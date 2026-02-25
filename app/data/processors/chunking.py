from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from app.utils.llm_utils import generate_summary
import re
import hashlib
from enum import Enum


class ChunkType(Enum):
    DOCUMENT = "doc"
    SECTION  = "sec"
    TEXT     = "txt"
    TABLE    = "tbl"


@dataclass
class Chunk:
    id:              str
    type:            ChunkType
    section:         str
    content:         str
    parent_id:       Optional[str]
    child_chunk_ids: List[str] = field(default_factory=list)
    summary:         Optional[str] = None
    metadata:        Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "section": self.section,
            "content": self.content,
            "parent_id": self.parent_id,
            "child_chunk_ids": self.child_chunk_ids,
            "summary": self.summary,
            "metadata": self.metadata
        }


class SECChunker:
    """
    Hierarchical SEC filing chunker.
    """
    # SEC 10-K standard sections
    SECTION_PATTERNS = {
            "Item 1": r"ITEM\s+1\.?\s*BUSINESS",
            "Item 1A": r"ITEM\s+1A\.?\s*RISK\s+FACTORS",
            "Item 1B": r"ITEM\s+1B\.?\s*UNRESOLVED\s+STAFF\s+COMMENTS",
            "Item 1C": r"ITEM\s+1C\.?\s*CYBERSECURITY",
            "Item 2": r"ITEM\s+2\.?\s*PROPERTIES",
            "Item 3": r"ITEM\s+3\.?\s*LEGAL\s+PROCEEDINGS",
            "Item 4": r"ITEM\s+4\.?\s*MINE\s+SAFETY",
            "Item 5": r"ITEM\s+5\.?\s*MARKET\s+FOR\s+REGISTRANT",
            "Item 6": r"ITEM\s+6\.?\s*SELECTED\s+FINANCIAL\s+DATA",
            "Item 7": r"ITEM\s+7\.?\s*MANAGEMENT\S\s+DISCUSSION\s+AND\s+ANALYSIS",
            "Item 7A": r"ITEM\s+7A\.?\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURE",
            "Item 8": r"ITEM\s+8\.?\s*FINANCIAL\s+STATEMENTS\s+AND\s+SUPPLEMENTARY\s+DATA",
            "Item 9": r"ITEM\s+9\.?\s*CHANGES\s+IN\s+AND\s+DISAGREEMENTS\s+WITH\s+ACCOUNTING",
            "Item 9A": r"ITEM\s+9A\.?\s*CONTROLS\s+AND\s+PROCEDURES",
            "Item 10": r"ITEM\s+10\.?\s*DIRECTORS,\s+EXECUTIVE\s+OFFICERS,\s+AND\s+CORPORATE\s+GOVERNANCE",
            "Item 11": r"ITEM\s+11\.?\s*EXECUTIVE\s+COMPENSATION",
            "Item 12": r"ITEM\s+12\.?\s*SECURITY\s+OWNER\s+RIGHTS",
            "Item 13": r"ITEM\s+13\.?\s*CERTAIN\s+RELATIONSHIPS\s+AND\s+RELATED\s+TRANSACTIONS",
            "Item 14": r"ITEM\s+14\.?\s*PRINCIPAL\s+ACCOUNTING\s+FEES\s+AND\s+SERVICES",
            "Item 15": r"ITEM\s+15\.?\s*EXHIBITS\s+AND\s+FINANCIAL\s+STATEMENT\s+SCHEDULES"
        }

    TABLE_PATTERN = r"\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n)+"

    def __init__(self, chunk_size: int = 1500, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def chunk_file(self, file_path: str, meta: Dict[str, Any]) -> List[Chunk]:
        """Read a file path and then chunk its contents."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.chunk(content, meta)

    def chunk(self, markdown: str, meta: Dict[str, Any]) -> List[Chunk]:
        """Parse a markdown SEC filing into a list of Chunks."""
        filing_id = meta["filing_id"]
        chunks: List[Chunk] = []

        # Level 0 – document root
        doc = self._doc_chunk(filing_id, meta)
        chunks.append(doc)

        # Level 1 + 2 – sections and their children
        for section, content in self._split_sections(markdown).items():
            sec   = self._section_chunk(filing_id, section, content, doc.id)
            doc.child_chunk_ids.append(sec.id)
            
            texts = self._text_chunks(filing_id, section, content, sec.id)
            for t in texts:
                t.summary = generate_summary(t.content)
                sec.child_chunk_ids.append(t.id)
                
            tables = self._table_chunks(filing_id, section, content, sec.id)
            for tbl in tables:
                tbl.summary = generate_summary(tbl.content)
                sec.child_chunk_ids.append(tbl.id)
                
            chunks += [sec, *texts, *tables]

        return chunks

    # ------------------------------------------------------------------ #
    # ID helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _slug(text: str) -> str:
        """
        Example: 'Item 1A' -> 'item_1a'
        """
        return re.sub(r"\s+", "_", text.strip().lower())

    @staticmethod
    def _short_hash(text: str) -> str:
        """
        Create unique hash for the text 
        """
        return hashlib.md5(text.encode()).hexdigest()[:6]

    def _make_id(self, chunk_type: ChunkType, filing_id: str,
                 *parts: str) -> str:
        """
        Produces readable, predictable IDs:
            doc_GOOGL10K2024
            sec_GOOGL10K2024_item_1a
            txt_GOOGL10K2024_item_7_0
            tbl_GOOGL10K2024_item_8_2
        """
        segments = [chunk_type.value, filing_id] + [p for p in parts if p]
        return "_".join(segments)

    # ------------------------------------------------------------------ #
    # Level 0 – document                                                 #
    # ------------------------------------------------------------------ #

    def _doc_chunk(self, filing_id: str, meta: Dict[str, Any]) -> Chunk:
        return Chunk(
            id        = self._make_id(ChunkType.DOCUMENT, filing_id),
            type      = ChunkType.DOCUMENT,
            section   = "document",
            content   = (
                f"{meta.get('company_name')} | {meta.get('filing_type')} | "
                f"CIK {meta.get('cik')} | FY {meta.get('fiscal_year')} | "
                f"Filed {meta.get('filing_date')}"
            ),
            parent_id = None,
            metadata  = {k: meta[k] for k in
                         ("company_name", "cik", "filing_type",
                          "filing_date", "fiscal_year") if k in meta},
        )

    # ------------------------------------------------------------------ #
    # Level 1 – sections                                                 #
    # ------------------------------------------------------------------ #

    def _section_chunk(self, filing_id: str, section: str,
                       content: str, parent_id: str) -> Chunk:
        slug = self._slug(section)
        return Chunk(
            id        = self._make_id(ChunkType.SECTION, filing_id, slug),
            type      = ChunkType.SECTION,
            section   = section,
            content   = content[:500],   # preview only – full text lives in txt chunks
            parent_id = parent_id,
            metadata  = {
                "char_count":   len(content),
                "has_tables":   bool(re.search(self.TABLE_PATTERN, content)),
                "is_risk":      "1a" in slug or "risk" in slug,
            },
        )

    # ------------------------------------------------------------------ #
    # Level 2 – text chunks (sliding window)                             #
    # ------------------------------------------------------------------ #

    def _text_chunks(self, filing_id: str, section: str,
                     content: str, parent_id: str) -> List[Chunk]:
        # Strip tables so text chunks contain only prose
        prose = re.sub(self.TABLE_PATTERN, "", content).strip()
        slug  = self._slug(section)

        sentences = re.split(r"(?<=[.!?])\s+", prose)
        windows   = self._sliding_windows(sentences)

        return [
            Chunk(
                id        = self._make_id(ChunkType.TEXT, filing_id, slug, str(i)),
                type      = ChunkType.TEXT,
                section   = section,
                content   = window,
                parent_id = parent_id,
                metadata  = {"chunk_index": i},
            )
            for i, window in enumerate(windows)
            if len(window) >= 100  # skip micro-chunks
        ]

    def _sliding_windows(self, sentences: List[str]) -> List[str]:
        """Build overlapping text windows from a sentence list."""
        windows, buf, buf_len = [], [], 0

        for sent in sentences:
            if buf_len + len(sent) > self.chunk_size and buf:
                windows.append(" ".join(buf))
                # keep overlap tail
                tail, tail_len = [], 0
                for s in reversed(buf):
                    if tail_len + len(s) <= self.overlap:
                        tail.insert(0, s)
                        tail_len += len(s)
                    else:
                        break
                buf, buf_len = tail, tail_len

            buf.append(sent)
            buf_len += len(sent)

        if buf:
            windows.append(" ".join(buf))
        return windows

    # ------------------------------------------------------------------ #
    # Level 2 – table chunks                                             #
    # ------------------------------------------------------------------ #

    def _table_chunks(self, filing_id: str, section: str,
                      content: str, parent_id: str) -> List[Chunk]:
        slug = self._slug(section)
        chunks = []

        for i, match in enumerate(re.finditer(self.TABLE_PATTERN, content)):
            table_md = match.group()
            lines    = [l.strip() for l in table_md.splitlines() if l.strip()]
            headers  = [c.strip() for c in lines[0].split("|") if c.strip()]

            chunks.append(Chunk(
                id        = self._make_id(ChunkType.TABLE, filing_id, slug, str(i)),
                type      = ChunkType.TABLE,
                section   = section,
                content   = table_md,
                parent_id = parent_id,
                metadata  = {
                    "headers":    headers,
                    "rows":       max(0, len(lines) - 2),
                    "is_financial": bool(re.search(
                        r"\$|revenue|income|assets|liabilities",
                        table_md, re.IGNORECASE)),
                },
            ))

        return chunks

    # ------------------------------------------------------------------ #
    # Section splitting                                                  #
    # ------------------------------------------------------------------ #

    def _split_sections(self, text: str) -> Dict[str, str]:
        matches = sorted(
            (
                (m.start(), m.end(), name)
                for name, pattern in self.SECTION_PATTERNS.items()
                for m in re.finditer(pattern, text, re.IGNORECASE)
            ),
            key=lambda x: x[0],
        )

        if not matches:
            return {"Full Document": text}

        sections = {}
        for i, (start, end, name) in enumerate(matches):
            next_start = matches[i + 1][0] if i + 1 < len(matches) else len(text)
            sections[name] = text[end:next_start].strip()

        if matches[0][0] > 0:
            sections["Preliminary"] = text[:matches[0][0]].strip()

        return sections


# --------------------------------------------------------------------------- #
# Quick demo                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json
    import os

    meta = {
        "filing_id":   "GOOGL10K2024",
        "company_name": "Google",
        "cik":         "0000320193",
        "filing_type": "10-K",
        "filing_date": "2024-10-27",
        "fiscal_year": 2024,
    }

    md_path = "/home/diwakar/dev/Fin-RAG/data/processed/GOOGL_10K_2024.md"
    chunker = SECChunker(chunk_size=1500, overlap=150)

    # Process file path directly
    print(f"--- Chunking {md_path} ---")
    chunks = chunker.chunk_file(md_path, meta)

    # Save to JSON
    output_path = "/home/diwakar/dev/Fin-RAG/data/processed/chunks_1_GOOGL10K2024.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2)

    print(f"Saved {len(chunks)} chunks to {output_path}")

    # Print first few for inspection
    for c in chunks[:5]:
        print(f"[{c.type.value}] {c.id}")
        print(f"  section  : {c.section}")
        print(f"  preview  : {c.content[:80].strip()!r}")
        print()