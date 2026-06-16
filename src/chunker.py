from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
# from langchain_core.documents import Document

from models import MinimalSource, ChunkData

def splitter(file_path: str, max_chunk_size: int) -> list[ChunkData]:
    # something like that:
        # splitter = RecursiveCharacterTextSplitter.from_language(
        #     chunk_size=300,
        #     chunk_overlap=0,
        #     language=Language.MARKDOWN,
        #     add_start_index=True
        # )

        # with open("../vllm-0.10.1/find_cuda_init.py") as f:
        #     text = f.read()

        # res = splitter.create_documents([text])

        # print(len(res))

        # for i, chunk in enumerate(res):
        #     start = chunk.metadata['start_index']
        #     end = chunk.metadata['start_index'] + len(chunk.page_content)
        #     is_valid = chunk.page_content == text[start:end]
        #     print(f"chunk {i}: valid={is_valid}")
        # return (res)

    pass

def get_text(doc: MinimalSource) -> ChunkData:
    # give the MinimalSource and return the str (need for the bm25)
    pass

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