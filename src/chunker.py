import pathlib
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .models import ChunkData


VALID_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml"}
SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".venv"}


def splitter_func(file_path: str, max_chunk_size: int) -> list[ChunkData]:
    """Splits a file into chunks based on its extension."""
    ext = pathlib.Path(file_path).suffix
    lang_map = {
        ".py": Language.PYTHON,
        ".md": Language.MARKDOWN,
        ".rst": Language.RST
        }

    if ext in lang_map:
        splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=max_chunk_size,
            chunk_overlap=max_chunk_size // 5,
            language=lang_map[ext],
            add_start_index=True
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=max_chunk_size // 5,
            add_start_index=True
        )
    try:
        with open(file_path) as f:
            text = f.read()
    except IOError:
        return []

    chunks = splitter.create_documents([text])
    result: list[ChunkData] = []
    for chunk in chunks:
        start = chunk.metadata["start_index"]
        end = start + len(chunk.page_content)
        result.append(ChunkData(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
            text=chunk.page_content
        ))
    return result


def loader(repo_path: str, max_chunk_size: int) -> list[ChunkData]:
    """Loads and chunks all valid files from a repository."""
    data_set: list[ChunkData] = []
    files = list(pathlib.Path(repo_path).rglob("*"))

    for f in files:
        if not f.is_file() or any(part in SKIP_DIRS for part in f.parts):
            continue
        if f.suffix.lower() not in VALID_EXTENSIONS:
            continue
        data_set.extend(splitter_func(str(f), max_chunk_size))

    return data_set
