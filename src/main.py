# # from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# # # splitter = RecursiveCharacterTextSplitter(
# # #     chunk_size=100,
# # #     chunk_overlap=0
# # # )

# # # with open("main.py", "r") as f:
# # #     text = f.read()

# # # res = splitter.create_documents([text])


# # # print(res)


# # # splitter = RecursiveCharacterTextSplitter.from_language(
# # #     chunk_size=2000,
# # #     chunk_overlap=200,
# # #     language=Language.PYTHON
# # #     )

# # # # is equel to:

# # # splitter = RecursiveCharacterTextSplitter(
# # #     chunk_size=2000,
# # #     chunk_overlap=200,
# # #     separator=["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]
# # #     )

# # # ___________________________________________________________________________________________________

# # # Exercise 1 — Basic observation:
# # # Take any .py file from the vLLM repo. Split it with chunk_size=500, chunk_overlap=0 using the generic splitter.
# # # Print each chunk with its index. Answer:

# # # How many chunks did you get?
# # # Did any chunk cut inside a function?

# # from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


# # splitter = RecursiveCharacterTextSplitter(
# #     chunk_size=500,
# #     chunk_overlap=0
# # )

# # with open("../vllm-0.10.1/find_cuda_init.py") as f:
# #     text = f.read()

# # res = splitter.create_documents([text])

# # print(len(res))

# # for i, chunk in enumerate(res):
# #     print(f"chunk={chunk}\n\n\n\tindex={i}\n\n")

# # # How many chunks did you get? 2

# # # Did any chunk cut inside a function? yes


# # # ___________________________________________________________________________________________________

# # # Exercise 2 — Generic vs Python-aware
# # # Same file, same chunk_size. Compare generic splitter vs from_language(Language.PYTHON). Answer:

# # # Where do the cuts differ?
# # # Which one respects def boundaries better?

# # from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


# # splitter = RecursiveCharacterTextSplitter.from_language(
# #     chunk_size=500,
# #     chunk_overlap=0,
# #     language=Language.PYTHON
# # )

# # with open("../vllm-0.10.1/find_cuda_init.py") as f:
# #     text = f.read()

# # res = splitter.create_documents([text])

# # print(len(res))

# # for i, chunk in enumerate(res):
# #     print(f"chunk={chunk}\n\n\n\tindex={i}\n\n")

# # # ___________________________________________________________________________________________________

# # # Exercise 3 — Overlap effect
# # # Same file, chunk_size=300. Run with chunk_overlap=0 then chunk_overlap=100. Answer:

# # # Do you see repeated text between consecutive chunks?
# # # Which version would give better retrieval context?


# # from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# # splitter = RecursiveCharacterTextSplitter(
# #     chunk_size=300,
# #     chunk_overlap=100
# # )

# # with open("../vllm-0.10.1/find_cuda_init.py") as f:
# #     text = f.read()

# # res = splitter.create_documents([text])

# # print(len(res))

# # for i, chunk in enumerate(res):
# #     print(f"chunk={chunk}\n\n\n\tindex={i}\n\n")

# # #this one is better

# # # ___________________________________________________________________________________________________

# # # Exercise 4 — Markdown splitting
# # # Take a .md file from vLLM docs. Split with from_language(Language.MARKDOWN). Answer:

# # # # Does each chunk start at a header?
# # # # What happens to a section longer than chunk_size?


# # from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# # splitter = RecursiveCharacterTextSplitter.from_language(
# #     chunk_size=50,
# #     chunk_overlap=10,
# #     language=Language.MARKDOWN
# # )

# # with open("../vllm-0.10.1/RELEASE.md") as f:
# #     text = f.read()

# # res = splitter.create_documents([text])

# # print(len(res))

# # for i, chunk in enumerate(res):
# #     print(f"chunk={chunk}\n\n\n\tindex={i}\n\n")

# # # What happens to a section longer than chunk_size? he just split it

# # # ___________________________________________________________________________________________________

# # # # Exercise 5 — Character index tracking
# # # # The real one. Take a file, split it, then for each chunk:

# # # # Find its start position in the original source string
# # # # Compute end = start + len(chunk)
# # # # Verify source[start:end] == chunk

# # # # This is exactly what your chunker module needs to do.

# # from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# # splitter = RecursiveCharacterTextSplitter.from_language(
# #     chunk_size=300,
# #     chunk_overlap=0,
# #     language=Language.MARKDOWN,
# #     add_start_index=True
# # )

# # with open("../vllm-0.10.1/find_cuda_init.py") as f:
# #     text = f.read()

# # res = splitter.create_documents([text])

# # print(len(res))

# # for i, chunk in enumerate(res):
# #     start = chunk.metadata['start_index']
# #     end = chunk.metadata['start_index'] + len(chunk.page_content)
# #     is_valid = chunk.page_content == text[start:end]
# #     print(f"chunk {i}: valid={is_valid}")



# import chunker

# data_set = chunker.loader("../vllm-0.10.1", 2000)

# print(len(data_set))
# # print(data_set[:2])

# corpus = []

# for data in data_set:
#     corpus.append(data.text)

# print(corpus[:2])

# import bm25s

# # Create your corpus here
# # corpus = [
# #     "a cat is a feline and likes to purr",
# #     "a dog is the human's best friend and loves to play",
# #     "a bird is a beautiful animal that can fly",
# #     "a fish is a creature that lives in water and swims",
# # ]

# corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

# # print(corpus_tokens)

# # Create the BM25 model and index the corpus
# retriever = bm25s.BM25()
# retriever.index(corpus_tokens)

# # Query the corpus
# query = "does the fish purr like a cat?"
# query_tokens = bm25s.tokenize(query)

# # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
# # To return docs instead of IDs, set the `corpus=corpus` parameter.
# results, scores = retriever.retrieve(query_tokens, k=2)

# for i in range(results.shape[1]):
#     doc, score = results[0, i], scores[0, i]
#     print(f"Rank {i+1} (score: {score:.2f}): {doc}")

# # You can save the arrays to a directory...
# retriever.save("animal_index_bm25")

# # You can save the corpus along with the model
# retriever.save("animal_index_bm25", corpus=corpus)

# # ...and load them when you need them
# import bm25s
# reloaded_retriever = bm25s.BM25.load("animal_index_bm25", load_corpus=True)
# # set load_corpus=False if you don't need the corpus


from indexer import indexer

indexer("../vllm-0.10.1", "ghire_tm_hh", 200)
