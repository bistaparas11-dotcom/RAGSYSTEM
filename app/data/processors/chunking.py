import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


class ContentType(Enum):
    METADATA = "metadata"
    SECTION_HEADER = "section_header"
    TEXT = "text"
    TABLE = "table"
    FINANCIAL_NOTE = "financial_note"
    RISK_FACTOR = "risk_factor"


@dataclass
class Chunk:
    id: str
    content: str
    content_type: ContentType
    section: str              # e.g., "Item 1. Business"
    sub_section: Optional[str]
    page_range: tuple
    summary: Optional[str]
    entities: List[Dict]      # Extracted entities
    parent_chunk_id: Optional[str]  # For hierarchical relationships
    child_chunk_ids: List[str]
    metadata: Dict[str, Any]
    embedding_vector: Optional[List[float]] = None


class SECHierarchicalChunker:
    def __init__(self):
        # SEC 10-K standard sections
        self.section_patterns = {
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

    def parse_markdown(self, markdown_text: str,
                       filing_metadata: Dict) -> List[Chunk]:
        """Parse markdown into hierarchical chunks"""
        chunks = []

        # Extract document-level metadata chunk
        meta_chunk = self._create_metadata_chunk(filing_metadata)
        chunks.append(meta_chunk)

        # Split by sections
        sections = self._split_by_sections(markdown_text)

        for section_name, section_content in sections.items():
            # Create section-level chunk
            section_chunk = self._create_section_chunk(
                section_name,
                section_content,
                parent_id=meta_chunk.id
            )
            chunks.append(section_chunk)

            # Further split into sub-chunks based on content type
            sub_chunks = self._split_section_content(
                section_content,
                section_name,
                parent_id=section_chunk.id
            )
            chunks.extend(sub_chunks)

        return chunks

    def _split_section_content(
            self, content: str, section_name: str, parent_id: str) -> List[Chunk]:
        """Intelligent splitting based on content patterns"""
        chunks = []

        # Detect tables (markdown format)
        table_pattern = r'(\|[^\n]+\|\n\|[-:\|\s]+\|\n(?:\|[^\n]+\|\n)+)'
        tables = re.finditer(table_pattern, content)

        last_end = 0
        for match in tables:
            # Text before table
            if match.start() > last_end:
                text_chunk = content[last_end:match.start()]
                if len(text_chunk.strip()) > 50:
                    chunks.append(self._create_text_chunk(
                        text_chunk, section_name, parent_id
                    ))

            # Table chunk
            table_content = match.group(1)
            chunks.append(self._create_table_chunk(
                table_content, section_name, parent_id
            ))
            last_end = match.end()

        # Remaining text
        if last_end < len(content):
            remaining = content[last_end:]
            if len(remaining.strip()) > 50:
                chunks.append(self._create_text_chunk(
                    remaining, section_name, parent_id
                ))

        return chunks

    def _create_table_chunk(self, table_md: str,
                            section: str, parent_id: str) -> Chunk:
        """Special handling for financial tables"""
        # Parse table structure
        rows = [row.strip()
                for row in table_md.strip().split('\n') if row.strip()]
        headers = [cell.strip() for cell in rows[0].split('|') if cell.strip()]

        # Extract table caption/context (usually preceding text)
        context = self._extract_table_context(table_md)

        return Chunk(
            id=f"table_{hash(table_md) % 10000000}",
            content=table_md,
            content_type=ContentType.TABLE,
            section=section,
            sub_section=None,
            page_range=(0, 0),  # Populate from OCR metadata
            summary=None,  # Will be generated
            entities=self._extract_table_entities(table_md, headers),
            parent_chunk_id=parent_id,
            child_chunk_ids=[],
            metadata={
                "table_headers": headers,
                "row_count": len(rows) - 2,  # Exclude header and separator
                "table_context": context,
                "is_financial_table": self._is_financial_table(headers)
            }
        )

    def _create_metadata_chunk(self, filing_metadata: Dict) -> Chunk:
        """Create a root chunk for document metadata"""
        return Chunk(
            id=f"doc_{filing_metadata.get('accession_number', 'unknown')}",
            content=str(filing_metadata),
            content_type=ContentType.METADATA,
            section="Document Metadata",
            sub_section=None,
            page_range=(0, 0),
            summary=None,
            entities=[],
            parent_chunk_id=None,
            child_chunk_ids=[],
            metadata=filing_metadata
        )

    def _split_by_sections(self, markdown_text: str) -> Dict[str, str]:
        """Split markdown into SEC sections using regex patterns"""
        sections = {}
        positions = []

        for name, pattern in self.section_patterns.items():
            match = re.search(pattern, markdown_text, re.IGNORECASE)
            if match:
                positions.append((match.start(), name))

        positions.sort()

        for i in range(len(positions)):
            start_pos, section_name = positions[i]
            end_pos = (positions[i + 1][0] if i + 1 < len(positions)
                       else len(markdown_text))
            sections[section_name] = markdown_text[start_pos:end_pos]

        return sections

    def _create_section_chunk(self, section_name: str,
                              content: str, parent_id: str) -> Chunk:
        """Create a chunk for a major document section"""
        return Chunk(
            id=f"section_{section_name.replace(' ', '_')}",
            content=content[:1000].strip(),  # Store intro/header part
            content_type=ContentType.SECTION_HEADER,
            section=section_name,
            sub_section=None,
            page_range=(0, 0),
            summary=None,
            entities=[],
            parent_chunk_id=parent_id,
            child_chunk_ids=[],
            metadata={"full_section_length": len(content)}
        )

    def _create_text_chunk(self, text: str,
                           section_name: str, parent_id: str) -> Chunk:
        """Create a chunk for a block of text"""
        return Chunk(
            id=f"text_{hash(text) % 10000000}",
            content=text.strip(),
            content_type=ContentType.TEXT,
            section=section_name,
            sub_section=None,
            page_range=(0, 0),
            summary=None,
            entities=[],
            parent_chunk_id=parent_id,
            child_chunk_ids=[],
            metadata={}
        )

    def _extract_table_context(self, table_md: str) -> str:
        """Heuristically extract table context (titles, headings)"""
        # Simplistic implementation: in production, this would look at
        # preceding lines in DOM/MD
        return "Financial Table"

    def _extract_table_entities(self, table_md: str,
                                headers: List[str]) -> List[Dict]:
        """Extract basic entities from table contents"""
        entities = []
        if "$" in table_md:
            entities.append({
                "type": "FINANCIAL_VALUE",
                "context": "currency_symbol_detected"
            })
        return entities

    def _is_financial_table(self, headers: List[str]) -> bool:
        """Identify if a table is financial (vs narrative) based on headers"""
        financial_keywords = {
            "balance", "income", "cash", "equity", "assets",
            "liabilities", "revenue", "net"
        }
        return any(any(kw in h.lower() for kw in financial_keywords)
                   for h in headers)
