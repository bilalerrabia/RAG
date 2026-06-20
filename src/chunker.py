"""Handles document chunking for the RAG pipeline."""
import pathlib
import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .models import ChunkData

VALID_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".sh"}
SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", "node_modules", ".venv", "build", "dist"}


def splitter_extensions_handler(file_extension: str, max_chunk_size: int) -> RecursiveCharacterTextSplitter:
    """Returns the appropriate text splitter based on file extension."""
    extension_map = {
        ".py": Language.PYTHON,
        ".md": Language.MARKDOWN,
        ".rst": Language.RST,
    }
    if file_extension in extension_map:
        return RecursiveCharacterTextSplitter.from_language(
            chunk_size=max_chunk_size,
            chunk_overlap=max_chunk_size // 5,
            language=extension_map[file_extension],
            add_start_index=True
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=max_chunk_size // 5,
        add_start_index=True
    )


def splitter_func(file_path: str, max_chunk_size: int) -> list[ChunkData]:
    """Splits a file into chunks based on its extension."""
    file_extension: str = pathlib.Path(file_path).suffix
    splitter = splitter_extensions_handler(file_extension, max_chunk_size)
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    res = splitter.create_documents([text])
    
    result: list[ChunkData] = []
    for chunk in res:
        start_index = chunk.metadata["start_index"]
        end_index = start_index + len(chunk.page_content)
        result.append(ChunkData(
            file_path=file_path,
            first_character_index=start_index,
            last_character_index=end_index,
            text=chunk.page_content,
        ))
    return result


def loader(repo_path: str, max_chunk_size: int) -> list[ChunkData]:
    """Loads and chunks all valid files from a repository."""
    data_set: list[ChunkData] = []
    files = list(pathlib.Path(repo_path).rglob("*"))
    for f in tqdm.tqdm(files, desc="Chunking files"):
        if not f.is_file():
            continue
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        if f.suffix.lower() not in VALID_EXTENSIONS:
            continue
        try:
            data_set.extend(splitter_func(str(f), max_chunk_size))
        except Exception:
            continue
    return data_set