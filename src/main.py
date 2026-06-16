from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

with open("main.py", "r") as f:
    text = f.read()

res = splitter.create_documents([text])


print(res)


splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=2000,
    chunk_overlap=200,
    language=Language.PYTHON
    )

# is equel to:

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separator=["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]
    )