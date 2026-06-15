from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

with open("main.py", "r") as f:
    text = f.read()

res = splitter.split_text(text=text)

for par in res:
    print(par, end="\n\n")