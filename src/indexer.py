from .chunker import loader
from .models import ChunkData
import bm25s
import tqdm
import json
import pathlib
import re
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

def preprocess_text(text: str) -> str:
    # Split snake_case and camelCase for BM25
    text = text.replace('_', ' ')
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return text.lower()

def indexer(repo_path: str, repo_to_save: str, max_chunk_size: int) -> None:
    data_set: list[ChunkData] = loader(repo_path, max_chunk_size)

    # 1. Prepare corpora
    raw_corpus = [data.text for data in data_set]
    processed_corpus = [preprocess_text(data.text) for data in tqdm.tqdm(data_set, desc="Processing corpus")]

    # 2. FAST ChromaDB Indexing (Pre-compute embeddings)
    client = chromadb.PersistentClient(path=repo_to_save)
    collection = client.get_or_create_collection(name="vllm_chunks")
    
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Pre-computing embeddings (fast mode)...")
    # This batches the embedding generation internally and is drastically faster
    embeddings = embedder.encode(raw_corpus, batch_size=128, show_progress_bar=True, convert_to_numpy=True)

    BATCH_SIZE = 500
    for i in tqdm.tqdm(range(0, len(data_set), BATCH_SIZE), desc="chromadb indexing"):
        batch = data_set[i : i + BATCH_SIZE]
        ids = [str(i + j) for j in range(len(batch))]
        metadatas = [
            {
                "file_path": chunk.file_path,
                "first_character_index": chunk.first_character_index,
                "last_character_index": chunk.last_character_index
            }
            for chunk in batch
        ]
        # Pass pre-computed embeddings to avoid Chroma's slow internal batching
        collection.add(
            ids=ids,
            documents=[chunk.text for chunk in batch],
            metadatas=metadatas,
            embeddings=embeddings[i : i + BATCH_SIZE].tolist()
        )

    # 3. Fast BM25 Indexing
    corpus_tokens = bm25s.tokenize(processed_corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    pathlib.Path(repo_to_save).mkdir(parents=True, exist_ok=True)
    retriever.save(f"{repo_to_save}/bm25_index")

    with open(f"{repo_to_save}/chunks.json", "w") as f:
        json.dump([chunk.model_dump() for chunk in data_set], f, indent=2)