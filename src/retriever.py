import bm25s
import json
import re
from .models import MinimalSource
from functools import lru_cache
from collections import Counter

@lru_cache(maxsize=256)
def get_file_content(file_path: str) -> str:
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        return f.read()

@lru_cache(maxsize=1)
def load_bm25_index(index_path: str):
    return bm25s.BM25.load(f"{index_path}/bm25_index")

@lru_cache(maxsize=1)
def load_chunks(chunks_path: str):
    with open(chunks_path, encoding="utf-8") as f:
        return json.load(f)

def preprocess_text(text: str) -> str:
    text = text.replace('_', ' ')
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return text.lower()

def expand_query_prf(query: str, bm25_index, chunks, top_n=3, terms=5) -> str:
    """
    MANDATORY Query Expansion Bonus: Pseudo-Relevance Feedback.
    Extracts top keywords from the initial search results to expand the query.
    """
    processed_query = preprocess_text(query)
    q_token = bm25s.tokenize([processed_query])
    
    # 1. Quick initial search
    results, _ = bm25_index.retrieve(q_token, k=top_n)
    
    # 2. Gather text from top chunks
    feedback_text = " ".join([chunks[idx]["text"] for idx in results[0]])
    feedback_text = preprocess_text(feedback_text)
    
    # 3. Extract most frequent words (excluding original query words)
    words = re.findall(r'\b[a-zA-Z0-9]+\b', feedback_text)
    counts = Counter(words)
    query_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', processed_query))
    
    expansion_terms = []
    for word, count in counts.most_common():
        if word not in query_words and len(word) > 2:
            expansion_terms.append(word)
        if len(expansion_terms) == terms:
            break
            
    if not expansion_terms:
        return query
        
    return query + " " + " ".join(expansion_terms)

def search_bm25(chunks_path: str, index_path: str, query: str, k: int) -> list[tuple[MinimalSource, float]]:
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
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]
import bm25s
import json
import re
from .models import MinimalSource
from functools import lru_cache
from collections import Counter

# 1. QUERY CACHE DICTIONARY
_QUERY_CACHE = {}

@lru_cache(maxsize=256)
def get_file_content(file_path: str) -> str:
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        return f.read()

@lru_cache(maxsize=1)
def load_bm25_index(index_path: str):
    return bm25s.BM25.load(f"{index_path}/bm25_index")

@lru_cache(maxsize=1)
def load_chunks(chunks_path: str):
    with open(chunks_path, encoding="utf-8") as f:
        return json.load(f)

# ... (keep preprocess_text, expand_query_prf, search_bm25, search_chromadb, normalize) ...

def hybrid_search(query: str, k: int, chunks_path, index_path, collection,
                   bm25_weight=0.6, chroma_weight=0.4) -> list[MinimalSource]:

    # 2. CHECK QUERY CACHE FIRST
    cache_key = (query, k)
    if cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]

    # --- MANDATORY QUERY EXPANSION (PRF) ---
    bm25_index = load_bm25_index(index_path)
    chunks = load_chunks(chunks_path)
    expanded_query = expand_query_prf(query, bm25_index, chunks)
    
    candidates = k * 5

    bm25_results = search_bm25(query=expanded_query, k=candidates, index_path=index_path, chunks_path=chunks_path)
    
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

    unique_chunks = []
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
        if len(unique_chunks) == k:
            break

    final_chunks = unique_chunks[:k]

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

    # 3. SAVE RESULT TO QUERY CACHE BEFORE RETURNING
    _QUERY_CACHE[cache_key] = final_sources

    return final_sources