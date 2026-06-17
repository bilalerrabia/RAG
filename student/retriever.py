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
from .models import MinimalSource, ChunkData




def retrieval(chunks_path: str, index_path: str, query: str, k: int) -> list[MinimalSource]:

    index = bm25s.BM25.load(f"{index_path}/bm25_index")

    results_list: list[MinimalSource] = []

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    q_token = bm25s.tokenize([query], stopwords="en")

    results, scores = index.retrieve(q_token, k=k)

    for rep_index in results[0]:
        chunkdata: ChunkData = chunks[rep_index]
        results_list.append(
            MinimalSource(
                file_path=chunkdata["file_path"],
                first_character_index=chunkdata["first_character_index"],
                last_character_index=chunkdata["last_character_index"]
        ))

    return results_list