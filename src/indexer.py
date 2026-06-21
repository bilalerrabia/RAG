"""Handles indexing of documents."""
from .chunker import loader
from .models import ChunkData
import bm25s
import tqdm
import json
import pathlib
import re
import chromadb
from sentence_transformers import SentenceTransformer

def preprocess_text(text: str) -> str:
    """Splits snake_case and camelCase to help BM25 understand code."""
    text = text.replace('_', ' ')
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return text.lower()

def indexer(repo_path: str, repo_to_save: str, max_chunk_size: int) -> None:
    """Indexes the repository using BM25 and ChromaDB."""
    data_set: list[ChunkData] = loader(repo_path, max_chunk_size)

    raw_corpus = [d.text for d in data_set]
    processed_corpus = [preprocess_text(d.text) for d in data_set]

    # 1. ChromaDB Indexing (Fast batched embeddings)
    client = chromadb.PersistentClient(path=repo_to_save)
    collection = client.get_or_create_collection(name="vllm_chunks")
    
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(raw_corpus, batch_size=128, show_progress_bar=True, convert_to_numpy=True)

    BATCH_SIZE = 500
    for i in tqdm.tqdm(range(0, len(data_set), BATCH_SIZE), desc="ChromaDB Indexing"):
        batch = data_set[i : i + BATCH_SIZE]
        collection.add(
            ids=[str(i + j) for j in range(len(batch))],
            documents=[d.text for d in batch],
            metadatas=[{"file_path": d.file_path, "first_character_index": d.first_character_index, "last_character_index": d.last_character_index} for d in batch],
            embeddings=embeddings[i : i + BATCH_SIZE].tolist()
        )

    # 2. BM25 Indexing
    corpus_tokens = bm25s.tokenize(processed_corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    pathlib.Path(repo_to_save).mkdir(parents=True, exist_ok=True)
    retriever.save(f"{repo_to_save}/bm25_index")

    with open(f"{repo_to_save}/chunks.json", "w", encoding="utf-8") as f:
        json.dump([d.model_dump() for d in data_set], f)