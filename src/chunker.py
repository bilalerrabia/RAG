from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
# from langchain_core.documents import Document
import pathlib
from models import MinimalSource, ChunkData

def splitter_func(file_path: str, max_chunk_size: int) -> list[ChunkData]:

    file_extension: str = pathlib.Path(file_path).suffix
    splitter = splitter_extensions_handler(file_extension, max_chunk_size)
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    res = splitter.create_documents([text])

    result: list[ChunkData] = []

    for chunk in res:
        start_index = chunk.metadata["start_index"]
        end_index = start_index + len(chunk.page_content)
        chunk.metadata["end_index"] = end_index

        result.append(ChunkData(
            file_path=file_path,
            first_character_index=start_index,
            last_character_index=end_index,
            text=chunk.page_content,
            ))

    return result

def splitter_extensions_handler(file_extension: str, max_chunk_size: int)-> RecursiveCharacterTextSplitter:

    extension = {
        ".py": Language.PYTHON,
        ".md": Language.MARKDOWN,
        ".rst": Language.RST,
    }

    if file_extention in extention:
        splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=max_chunk_size,
        chunk_overlap=max_chunk_size // 10,
        language=extension[file_extension],
        add_start_index=True
        )

    else:
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=max_chunk_size // 10,
        add_start_index=True
        )

    return splitter

def loader(repo_path : str, max_chunk_size: int)-> list[ChunkData]:
    # will start read the files from vllm-0.10.1
    # and call the splitter_extensions_handler to get the right split strategy then call the splitter
    pass