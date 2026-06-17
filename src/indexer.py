from .chunker import loader
from .models import ChunkData
import bm25s
import tqdm
import json
import pathlib


def indexer(repo_path: str, repo_to_save: str, max_chunk_size: int) -> None:

    data_set: list[ChunkData] = loader(repo_path, max_chunk_size)

    corpus: list[str] = [
        data.text for data in tqdm.tqdm(data_set, desc="Building corpus")
    ]

    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    pathlib.Path(repo_to_save).mkdir(parents=True, exist_ok=True)
    retriever.save(repo_to_save)

    with open(f"{repo_to_save}/chunks.json", "w") as f:

        json.dump([chunk.model_dump() for chunk in data_set], f, indent=2)
