*This project has been created as part of the 42 curriculum by berrabia.*

# RAG against the Machine

## Description

RAG against the Machine is a Retrieval-Augmented Generation (RAG) system designed to answer natural language questions about the `vLLM` codebase. Large Language Models often hallucinate or lack up-to-date knowledge about specific codebases. Instead of retraining a model, this project gives the LLM "external memory" by indexing the repository and retrieving the most relevant code and documentation snippets for a given query.

The system ingests the codebase, chunks the files using language-aware boundaries, and builds a dual index (BM25 and ChromaDB). When a question is asked, a hybrid retrieval system finds the best matches, feeds them as context to a local `Qwen/Qwen3-0.6B` model, and generates an accurate, source-grounded answer. The pipeline's retrieval quality is measured using the `recall@k` metric, requiring a 5% character overlap to count as a valid match.

---

## System Architecture

The RAG pipeline consists of four main components that interact seamlessly:

### 1. Ingestion & Indexing System
Reads the repository, chunks files based on their language (Python, Markdown), and builds dual indexes: a keyword-based BM25 index and a semantic ChromaDB vector index. Embeddings are pre-computed in batches to optimize indexing time.

### 2. Retrieval System
Takes a user query, searches both indexes in parallel, normalizes and blends the scores (Hybrid Search), deduplicates overlapping chunks, and applies a Cross-Encoder re-ranker to push the most relevant chunks to the top.

### 3. Answer Generation System
Extracts the text from the retrieved sources, truncates it to fit token limits, and passes it as context to the `Qwen/Qwen3-0.6B` LLM to generate a concise answer.

### 4. Evaluation System
Compares the retrieved sources against a ground-truth dataset using the `recall@k` metric, calculating the overlap between retrieved and correct sources.

---

## Chunking Strategy

Document segmentation is handled by LangChain's `RecursiveCharacterTextSplitter`.

*   **Language-Aware Splitting**: Python, Markdown, and ReStructuredText files are split using language-specific separators (e.g., class/function boundaries for Python, headers for Markdown) to preserve logical blocks.
*   **Chunk Size**: The maximum chunk size is strictly capped at 2000 characters (configurable via CLI).
*   **Overlap**: A 400-character overlap (`max_chunk_size // 5`) is used to prevent splitting crucial function signatures or paragraphs in half.
*   **Filtering**: Non-code files (images, binaries, `.git` directories) are explicitly filtered out to maintain index quality and reduce noise.

---

## Retrieval Method

We implement a **Hybrid Search** mechanism combining lexical and semantic search, followed by a **Cross-Encoder Re-ranker**:

1.  **BM25 (Lexical)**: We preprocess text and queries by splitting `snake_case` and `camelCase` words. This allows BM25 to match specific code identifiers (e.g., turning `getMetrics` into `get metrics`).
2.  **ChromaDB (Semantic)**: We use `all-MiniLM-L6-v2` to embed chunks. ChromaDB receives the raw, unmodified query to understand the natural language intent.
3.  **Score Blending**: BM25 scores and ChromaDB distances are normalized to a `[0, 1]` range. They are then combined using a weighted average (60% BM25, 40% ChromaDB).
4.  **Re-ranking**: The top 15 unique candidates are passed to `ms-marco-MiniLM-L-6-v2` to deeply compare the query and the chunk. The re-ranker score is blended 50/50 with the hybrid score to protect exact code matches.
5.  **Smart Expansion**: The boundaries of the final top-k results are expanded to exactly 2000 characters to maximize the overlap ratio with ground-truth sources without breaking the validator's length limit.

---

## Performance Analysis

*   **Recall@5**: Achieves **>85% on documentation** questions and **>55% on code** questions, comfortably passing the 80% and 50% thresholds respectively.
*   **Indexing Time**: Completes in **< 280 seconds**, well under the 5-minute limit. This is achieved by pre-computing embeddings in batches of 128 rather than using ChromaDB's internal, slower batching.
*   **Warm Retrieval Throughput**: 200 questions are retrieved in **< 40 seconds**, far exceeding the 90-second limit. This is made possible by the 3-tier caching system and the fast MiniLM re-ranker.

---

## Design Decisions

*   **Dual Indexing**: Relying solely on embeddings misses exact code identifiers, while relying solely on BM25 misses synonyms. Hybrid search provides the best of both worlds.
*   **3-Tier Caching System (Bonus)**:
    1.  *Index Caching*: BM25 and ChromaDB indexes are loaded into RAM once using `@lru_cache`.
    2.  *Resource Caching*: File contents are cached in RAM so that reading `metrics.py` 10 times only hits the disk once.
    3.  *Query Caching*: A dictionary stores results of previous queries, returning them in `O(1)` time if duplicated.
*   **Lazy LLM Loading**: The LLM is only loaded into memory when the `answer` or `answer_dataset` commands are executed, keeping the cold-start latency for search-only operations under 60 seconds.

---

## Challenges Faced

*   **Strict 2000-character Limit**: The moulinette rejects sources longer than 2000 characters. Our initial "whole-file" approach failed validation. We solved this by implementing "Smart Expansion" to expand chunks exactly to the 2000-char boundary.
*   **Slow ChromaDB Indexing**: Initial indexing took >5 minutes. We bypassed this by pre-computing embeddings with `SentenceTransformer` and passing them directly to ChromaDB.
*   **camelCase Tokenization**: BM25 failed to match code queries because it treated `BaseProcessingInfo` as a single unknown word. We solved this with a regex preprocessor that splits camelCase and snake_case before tokenization.

---

## Instructions

### Prerequisites
*   Python 3.10+
*   `uv` package manager

### Installation
Clone the repository and install dependencies:
```bash
uv sync
```

### Makefile Commands
*   `make install`: Install dependencies.
*   `make run`: Execute the main CLI.
*   `make lint`: Run `flake8` and `mypy` type checking.
*   `make clean`: Remove temporary caches (`__pycache__`, `.mypy_cache`).

---

## Example Usage

### 1. Index the repository
```bash
uv run python -m src index --repo_path data/raw/vllm-0.10.1 --max_chunk_size 2000
```

### 2. Search a single query
```bash
uv run python -m src search --query "How to configure OpenAI server?" --k 10
```

### 3. Process a dataset of questions
```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10
```

### 4. Generate answers for the dataset
```bash
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json
```

### 5. Evaluate search results
```bash
uv run python -m src evaluate --student_path data/output/search_results/dataset_docs_public.json --right_answers_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

---

## Resources & AI Usage

### Resources
*   [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
*   [ChromaDB Documentation](https://docs.trychroma.com/)
*   [BM25S Documentation](https://github.com/xhluca/bm25s)
*   [LangChain Text Splitters](https://python.langchain.com/docs/how_to/splitter/)

https://www.coursera.org/learn/retrieval-augmented-generation-rag

https://youtu.be/reAmcocQyBA

https://www.youtube.com/watch?v=Qs_y0lTJAp0&list=PL84IF1fUunhNtKdA0-j8ITM6HC3F6QG-t

https://youtu.be/8OJC21T2SL4




### AI Usage
AI tools (ChatGPT / Claude) were utilized during development to:
*   Debug Pydantic validation errors when interfacing with the moulinette.
*   Optimize HuggingFace `generate` kwargs (e.g., discovering `@torch.inference_mode` and `enable_thinking=False` for Qwen).
*   Conceptualize the regex pattern required to split `camelCase` identifiers for the BM25 tokenizer.
*   Architect the 3-tier caching system to ensure the 90-second throughput limit was safely met.