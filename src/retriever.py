# load BM25 index from disk
# load chunks.json from disk
#           ↓
# receive a query string + k
#           ↓
# tokenize the query with bm25s.tokenize()
#           ↓
# retriever.retrieve() → top-k chunk indexes + scores
#           ↓
# map indexes → ChunkData objects
#           ↓
# convert ChunkData → MinimalSource
#           ↓
# return list[MinimalSource]
import bm25s
import json
from functools import lru_cache
from .models import MinimalSource, ChunkData

# Cache these so they are only loaded from disk ONCE
@lru_cache(maxsize=1)
def load_bm25_index(index_path: str):
    return bm25s.BM25.load(f"{index_path}/bm25_index")

@lru_cache(maxsize=1)
def load_chunks(chunks_path: str):
    with open(chunks_path, encoding="utf-8") as f:
        return json.load(f)

def search_bm25(chunks_path: str, index_path: str, query: str, k: int) -> list[tuple[MinimalSource, float]]:
    # Load from memory (or disk if first time)
    index = load_bm25_index(index_path)
    chunks = load_chunks(chunks_path)

    q_token = bm25s.tokenize([query])

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

# ... keep search_chromadb and hybrid_search the same ...

def search_chromadb(collection, query: str, k: int) -> list[tuple[MinimalSource, float]]:
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    sources: list[tuple[MinimalSource, float]] = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        source = MinimalSource(
            file_path=metadata["file_path"],
            first_character_index=metadata["first_character_index"],
            last_character_index=metadata["last_character_index"]
        )
        sources.append((source, float(distance)))

    return sources

# def search_bm25(chunks_path: str, index_path: str, query: str, k: int) -> list[tuple[MinimalSource, float]]:

#     index = bm25s.BM25.load(f"{index_path}/bm25_index")

#     results_list: list[tuple[MinimalSource, float]] = []

#     with open(chunks_path, encoding="utf-8") as f:
#         chunks = json.load(f)

#     q_token = bm25s.tokenize([query])

#     results, scores = index.retrieve(q_token, k=k)

#     for i, rep_index in enumerate(results[0]):
#         chunkdata = chunks[rep_index]
#         source = MinimalSource(
#             file_path=chunkdata["file_path"],
#             first_character_index=chunkdata["first_character_index"],
#             last_character_index=chunkdata["last_character_index"]
#         )
#         results_list.append((source, float(scores[0][i])))

#     return results_list


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]

def hybrid_search(query: str, k: int, chunks_path, index_path, collection,
                bm25_weight=0.8, chroma_weight=0.2) -> list[MinimalSource]:

    # Fetch 10x more candidates to ensure we find enough unique files
    candidates = k * 10

    bm25_results = search_bm25(query=query, k=candidates, index_path=index_path, chunks_path=chunks_path)
    chroma_results = search_chromadb(collection=collection, query=query, k=candidates)

    bm25_raw = [score for (_, score) in bm25_results]
    bm25_norm = normalize(bm25_raw)

    chroma_raw = [score for (_, score) in chroma_results]
    chroma_norm = [1 - s for s in normalize(chroma_raw)]

    score_map = {}
    source_map = {}  # Stores one chunk per file to fallback on if file reading fails

    # Combine scores, but DEDUPLICATE by file_path
    for (source, _), norm_score in zip(bm25_results, bm25_norm):
        key = source.file_path
        score = bm25_weight * norm_score
        if key not in score_map or score > score_map[key]:
            score_map[key] = score
            source_map[key] = source

    for (source, _), norm_score in zip(chroma_results, chroma_norm):
        key = source.file_path
        score = chroma_weight * norm_score
        if key in score_map:
            score_map[key] += score
        else:
            score_map[key] = score
            source_map[key] = source

    # Sort files by their combined hybrid score
    sorted_files = sorted(score_map, key=lambda k: score_map[k], reverse=True)

    final_sources = []
    for file_path in sorted_files[:k]:
        try:
            # Expand the source to cover the ENTIRE file
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            file_len = len(content)
            
            final_sources.append(MinimalSource(
                file_path=file_path,
                first_character_index=0,
                last_character_index=file_len
            ))
        except Exception:
            # Fallback to chunk indices if the file can't be read
            orig = source_map[file_path]
            final_sources.append(MinimalSource(
                file_path=file_path,
                first_character_index=orig.first_character_index,
                last_character_index=orig.last_character_index
            ))

    return final_sources