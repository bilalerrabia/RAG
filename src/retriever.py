"""Handles retrieval of documents."""
import bm25s
import json
import re
from typing import Any
from .models import MinimalSource
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection


@lru_cache()
def load_bm25_index(index_path: str) -> bm25s.BM25:
    return bm25s.BM25.load(f"{index_path}/bm25_index")


@lru_cache()
def load_chunks(chunks_path: str) -> Any:
    with open(chunks_path) as f:
        return json.load(f)


@lru_cache()
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer('all-MiniLM-L6-v2')


def preprocess_text(text: str) -> str:
    text = text.replace('_', ' ')
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return text.lower()


def search_bm25(
    chunks_path: str,
    index_path: str,
    query: str,
    k: int
) -> list[tuple[MinimalSource, float]]:
    collection = load_bm25_index(index_path)
    chunks = load_chunks(chunks_path)

    processed_query = preprocess_text(query)
    q_token = bm25s.tokenize([processed_query])
    results, scores = collection.retrieve(q_token, k=k)

    results_list: list[tuple[MinimalSource, float]] = []
    for i, rep_index in enumerate(results[0]):
        chunkdata = chunks[rep_index]
        source = MinimalSource(
            file_path=chunkdata["file_path"],
            first_character_index=chunkdata["first_character_index"],
            last_character_index=chunkdata["last_character_index"]
        )
        results_list.append((source, float(scores[0][i])))
    return results_list


def search_chromadb(
    collection: Collection,
    query: str, k: int
) -> list[tuple[MinimalSource, float]]:
    embedder = get_embedder()
    query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=k)
    sources: list[tuple[MinimalSource, float]] = []
    for metadata, distance in zip(
            results["metadatas"][0], results["distances"][0]):
        source = MinimalSource(
            file_path=metadata["file_path"],
            first_character_index=metadata["first_character_index"],
            last_character_index=metadata["last_character_index"]
        )
        sources.append((source, float(distance)))
    return sources


def mini_softMax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_search(
    query: str,
    k: int,
    chunks_path: str,
    index_path: str,
    collection: Collection,
    bm25_weight: float = 0.8,
    chroma_weight: float = 0.2
) -> list[MinimalSource]:
    """Performs simple hybrid search and merges results."""

    # 1. Fetch from both retrievers
    bm25_results = search_bm25(
        query=query, k=k*5, index_path=index_path,
        chunks_path=chunks_path)
    chroma_results = search_chromadb(collection=collection, query=query, k=k*5)

    bm25_norm = mini_softMax([score for (_, score) in bm25_results])
    chroma_norm = [1 - s for s in mini_softMax(
        [score for (_, score) in chroma_results])]

    # 2. Merge into a dictionary (key -> [score, source])
    merged = {}

    for (source, _), score in zip(bm25_results, bm25_norm):
        key = (
            source.file_path,
            source.first_character_index,
            source.last_character_index
            )
        merged[key] = [bm25_weight * score, source]

    for (source, _), score in zip(chroma_results, chroma_norm):
        key = (
            source.file_path,
            source.first_character_index,
            source.last_character_index
            )
        if key in merged:
            merged[key][0] += chroma_weight * score
        else:
            merged[key] = [chroma_weight * score, source]

    # 3. Sort by highest combined score
    sorted_items = sorted(merged.values(), key=lambda x: x[0], reverse=True)

    # 4. Return the top k source objects directly
    final_sources = [item[1] for item in sorted_items[:k]]

    return final_sources
