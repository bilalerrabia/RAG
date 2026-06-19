from .chunker import loader
from .models import ChunkData
import bm25s
import tqdm
import json
import pathlib
import chromadb




def index_chromadb(data_set: list[ChunkData], save_path: str):

    client = chromadb.PersistentClient(path=save_path)
    collection = client.get_or_create_collection(name="vllm_chunks")

    BATCH_SIZE = 500

    for i in tqdm.tqdm(range(0, len(data_set), BATCH_SIZE), desc="chromadb indexing"):
        batch = data_set[i : i + BATCH_SIZE]

        ids       = [str(i + j) for j in range(len(batch))]
        documents = [chunk.text for chunk in batch]
        metadatas = [
            {
                "file_path": chunk.file_path,
                "first_character_index": chunk.first_character_index,
                "last_character_index": chunk.last_character_index
            }
            for chunk in batch
        ]
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    print("ChromaDB index saved")

    return collection



def indexer(repo_path: str, repo_to_save: str, max_chunk_size: int) -> None:

    data_set: list[ChunkData] = loader(repo_path, max_chunk_size)

    corpus: list[str] = [
        data.text for data in tqdm.tqdm(data_set, desc="Building corpus")
    ]

    index_chromadb(data_set, repo_to_save)

    corpus_tokens = bm25s.tokenize(corpus)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    pathlib.Path(repo_to_save).mkdir(parents=True, exist_ok=True)
    retriever.save(f"{repo_to_save}/bm25_index")

    with open(f"{repo_to_save}/chunks.json", "w") as f:

        json.dump([chunk.model_dump() for chunk in data_set], f, indent=2)
