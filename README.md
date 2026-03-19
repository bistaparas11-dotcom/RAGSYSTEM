# FinRAG: SEC 10-K Financial RAG

FinRAG is a hybrid RAG system specifically designed for analyzing SEC 10-K financial filings. It combines the power of knowledge graphs (Neo4j) and vector databases (Weaviate).

## Architecture

FinRAG is inspired by LightRAG but has simple implementation and uses vector search as well.

- **RAG Engine (`FinRAGEngine`)**: The core orchestrator that manages conversation history, resolves filing IDs, and coordinates retrieval and generation.
- **Retrieval Layer (`HybridRAGRetriever`)**: Uses a "HybridRAG" approach, blending:
  - **Vector Retrieval (Weaviate)**: Semantic search over text chunks.
  - **Graph Retrieval (Neo4j)**: Traverses relationships between entities, filings, and chunks for structured context.
  - **Reranking**: Refines results using [rerank-qa-mistral-4b](https://build.nvidia.com/nvidia/rerank-qa-mistral-4b?snippet_tab=Python) for maximum relevance.
- **Storage Layer**:
  - **Neo4j**: Stores the Knowledge Graph (Entities, Relationships, Filing Metadata, Summary, Chunk content).
  - **Weaviate**: Stores dense vector embeddings for semantic search.
- **Processing Pipeline**:
  - **OCR Layer**: Converts SEC PDFs to structured Markdown using PaddleOCR-VL.
  - **Enrichment Layer**: Automatically extracts entities and relationships using LLM-based analysis.

## Program Flow

The system operates in two main phases: **Ingestion** and **Query Execution**.

### 1. Ingestion Pipeline
1. **Download**: `SECBulkDownloader` fetches 10-K filings from SEC EDGAR.
2. **Convert**: `process_pdf` uses PaddleOCR-VL to transform complex PDFs into clean Markdown.
3. **Chunk**: `FastChunkPipeline` performs structural chunking based on document headers.
4. **Enrich**: DeepSeek-R1 analyzes each chunk on Modal to extract summary, key entities, financial metrics, and relationships.
5. **Load**: 
   - `KnowledgeGraph` builds a multi-relational graph in Neo4j AuraDB.
   - `WeaviateIngestor` generates and stores embeddings generated using [bge-m3](https://build.nvidia.com/baai/bge-m3?snippet_tab=Python) in Weaviate.

### 2. Query Execution (RAG Flow)
1. **Input**: User asks a question.
2. **Resolve**: `FilingResolver` identifies the relevant companies/filings mentioned.
3. **Retrieve**: `HybridRAGRetriever` performs a hybrid search:
   - **Local Search**: Fetches specific chunks and entities directly related to the query.
   - **Global Search**: Traverses the graph to find broader connections and cross-company entities.
4. **Rerank**: The most relevant context is selected and prioritized.
5. **Generate**: A prompt is built with the retrieved context and sent to Groq.

## Commands

### Ingestion
```bash
python -m app.data.processors.ingestion_manager
```

### Query

- Chatbot CLI
  ```bash
  python -m app.chatbot
  ```

- Chatbot Gradio UI
  ```bash
  python -m app.gradio_app
  ```

## Tech Stack

- **LLM**: Groq API
- **Embeddings**: [bge-m3](https://build.nvidia.com/baai/bge-m3?snippet_tab=Python)
- **Reranker**: [rerank-qa-mistral-4b](https://build.nvidia.com/nvidia/rerank-qa-mistral-4b?snippet_tab=Python)
- **Databases**: Neo4j AuraDB (Graph), Weaviate (Vector)
- **UI**: Gradio, CLI
- **OCR**: PaddleOCR-VL
- **Compute**: Modal, Nvidia NIM
