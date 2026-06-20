import bm25s
import json
import re
import torch
from .models import MinimalSource
from functools import lru_cache
from sentence_transformers import CrossEncoder
from collections import Counter

# ==========================================================
# BONUS: 3-TIER CACHING SYSTEM
# ==========================================================
_QUERY_CACHE: dict = {}

@lru_cache(maxsize=256)
def get_file_content(file_path: str) -> str:
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        return f.read()

@lru_cache(maxsize=1)
def load_bm25_index(index_path: str) -> bm25s.BM25:
    return bm25s.BM25.load(f"{index_path}/bm25_index")

@lru_cache(maxsize=1)
def load_chunks(chunks_path: str) -> list:
    with open(chunks_path, encoding="utf-8") as f:
        return json.load(f)

# Load the fast re-ranker
@lru_cache(maxsize=1)
def get_reranker():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device, max_length=512)
# ==========================================================

def preprocess_text(text: str) -> str:
    """Preprocesses text by splitting snake_case and camelCase."""
    text = text.replace('_', ' ')
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return text.lower()

def search_bm25(chunks_path: str, index_path: str, query: str, k: int) -> list[tuple[MinimalSource, float]]:
    """Searches the BM25 index."""
    index = load_bm25_index(index_path)
    chunks = load_chunks(chunks_path)
    processed_query = preprocess_text(query)
    q_token = bm25s.tokenize([processed_query])
    results, scores = index.retrieve(q_token, k=k)

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

def search_chromadb(collection, query: str, k: int) -> list[tuple[MinimalSource, float]]:
    """Searches the ChromaDB index."""
    results = collection.query(query_texts=[query], n_results=k)
    sources: list[tuple[MinimalSource, float]] = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        source = MinimalSource(
            file_path=metadata["file_path"],
            first_character_index=metadata["first_character_index"],
            last_character_index=metadata["last_character_index"]
        )
        sources.append((source, float(distance)))
    return sources

def normalize(scores: list[float]) -> list[float]:
    """Normalizes a list of scores to range [0, 1]."""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]
def hybrid_search(query: str, k: int, chunks_path: str, index_path: str, collection,
                   bm25_weight: float = 0.6, chroma_weight: float = 0.4) -> list[MinimalSource]:
    """Performs hybrid search using BM25 and ChromaDB, followed by Cross-Encoder Re-ranking."""
    # --- TIER 1: QUERY CACHE CHECK ---
    cache_key = (query, k)
    if cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]

    candidates = k * 5
    bm25_results = search_bm25(query=query, k=candidates, index_path=index_path, chunks_path=chunks_path)
    
    try:
        chroma_results = search_chromadb(collection=collection, query=query, k=candidates)
        chroma_norm = [1 - s for s in normalize([score for (_, score) in chroma_results])]
    except Exception:
        chroma_results = []
        chroma_norm = []

    bm25_norm = normalize([score for (_, score) in bm25_results])

    score_map = {}
    source_map = {}
    for (source, _), norm_score in zip(bm25_results, bm25_norm):
        key = (source.file_path, source.first_character_index, source.last_character_index)
        score_map[key] = bm25_weight * norm_score
        source_map[key] = source

    for (source, _), norm_score in zip(chroma_results, chroma_norm):
        key = (source.file_path, source.first_character_index, source.last_character_index)
        if key in score_map:
            score_map[key] += chroma_weight * norm_score
        else:
            score_map[key] = chroma_weight * norm_score
            source_map[key] = source

    sorted_chunks = sorted(score_map, key=lambda k: score_map[k], reverse=True)

    # Deduplicate overlapping chunks
    unique_chunks: list = []
    for key in sorted_chunks:
        source = source_map[key]
        is_dup = False
        for existing in unique_chunks:
            if existing.file_path == source.file_path:
                overlap_start = max(source.first_character_index, existing.first_character_index)
                overlap_end = min(source.last_character_index, existing.last_character_index)
                if overlap_end - overlap_start > 500:
                    is_dup = True
                    break
        if not is_dup:
            unique_chunks.append(source)
        if len(unique_chunks) == 15:  # Take top 15 unique for re-ranking
            break

    # --- CROSS-ENCODER RE-RANKING ---
    rerank_candidates = unique_chunks[:15]
    reranker = get_reranker()
    pairs = []
    valid_sources = []
    
    for source in rerank_candidates:
        try:
            content = get_file_content(source.file_path)
            # Pass 512 chars to the re-ranker for speed
            text = content[source.first_character_index:source.last_character_index][:512]
            pairs.append((query, text))
            valid_sources.append(source)
        except Exception:
            continue
            
    if pairs:
        r_scores = reranker.predict(pairs, batch_size=16)
        min_r = min(r_scores)
        max_r = max(r_scores)
        if max_r > min_r:
            norm_r = [(r - min_r) / (max_r - min_r) for r in r_scores]
        else:
            norm_r = [1.0 for _ in r_scores]
            
        blended = []
        for source, r_score in zip(valid_sources, norm_r):
            h_score = score_map.get((source.file_path, source.first_character_index, source.last_character_index), 0)
            # 50% Re-ranker + 50% Hybrid (Protects code chunks!)
            final_score = 0.5 * r_score + 0.5 * h_score
            blended.append((source, final_score))
            
        blended.sort(key=lambda x: x[1], reverse=True)
        final_chunks = [s for s, _ in blended[:k]]
    else:
        final_chunks = rerank_candidates[:k]

    # Smart Expansion to 2000 chars
    final_sources = []
    for source in final_chunks:
        content = get_file_content(source.file_path)
        file_len = len(content)
        first = source.first_character_index
        last = source.last_character_index
        
        if last - first < 2000:
            room = 2000 - (last - first)
            expand_left = room // 2
            expand_right = room - expand_left
            new_first = max(0, first - expand_left)
            new_last = min(file_len, last + expand_right)
            if new_first == 0:
                new_last = min(file_len, new_last + (expand_left - first))
            if new_last == file_len:
                new_first = max(0, new_first - (expand_right - (file_len - last)))
            first = new_first
            last = new_last
        
        final_sources.append(MinimalSource(
            file_path=source.file_path,
            first_character_index=first,
            last_character_index=last
        ))

    # --- TIER 1: QUERY CACHE STORE ---
    _QUERY_CACHE[cache_key] = final_sources
    return final_sources