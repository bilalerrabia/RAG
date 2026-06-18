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



# # Step 1 - test chunker alone first
# chunks = loader("vllm-0.10.1", 500)
# print(f"Total chunks: {len(chunks)}")
# print(f"First chunk: {chunks[0]}")

# # Step 2 - build index
# indexer("vllm-0.10.1", "data/processed", 500)
# print("Indexing done")

# # Step 3 - test retrieval
# results = retrieval(
#     chunks_path="data/processed/chunks.json",   # where is chunks.json?
#     index_path="data/processed",    # where did indexer save the BM25 index?
#     query="how does vLLM handle memory?",
#     k=5
# )

import fire
import tqdm
import json
import pathlib
from .chunker import loader
from .models import MinimalSearchResults, StudentSearchResults, StudentSearchResultsAndAnswer, AnsweredQuestion
from .indexer import indexer
from .retriever import retrieval
from .evaluator import evaluate
from .answerer import answerer

# index          → index the repository                            <- done
# search         → search for a single query                       <- done
# search_dataset → process multiple questions from JSON            <- done
# answer         → answer a single question with context           <- done
# answer_dataset → generate answers from search results            <- done
# evaluate       → evaluate search results against ground truth    <- done


class RAG:

    def index(self, repo_path="data/raw/vllm-0.10.1", repo_to_save="data/processed", max_chunk_size=2000) -> None:

        indexer(repo_path, repo_to_save, max_chunk_size)
        print(f"Ingestion complete! Indices saved under {repo_to_save}")

    def search(self,
    chunks_path: str = "data/processed/chunks.json", index_path: str = "data/processed",
    query: str = "", k: int = 5) -> list[MinimalSource]:

        results: list[MinimalSource] = retrieval(
            chunks_path=chunks_path,
            index_path=index_path,
            query=query,
            k=k
        )

        return results

    def search_dataset(
            self, dataset_path: str="data/datasets/UnansweredQuestions/dataset_docs_public.json",
            k:int = 10, save_directory: str = "data/output/search_results"):

        filename = pathlib.Path(dataset_path).name


        answers: list[MinimalSearchResults] = []
        with open(dataset_path) as f:
            questions = json.load(f)


        for q in tqdm.tqdm(questions["rag_questions"], desc="searching"):
            answers.append(MinimalSearchResults(
                question_id=q["question_id"],
                question=q["question"],
                retrieved_sources=self.search(query=q["question"], k=k)
                ))

        pathlib.Path(save_directory).mkdir(parents=True, exist_ok=True)

        res: StudentSearchResults = StudentSearchResults(
            k=k,
            search_results=answers
            )

        with open(f'{save_directory}/{filename}', "w") as f:
            json.dump(res.model_dump(), f, indent=4)

        print(f"Saved student_search_results to {save_directory}/{filename}")


    def evaluate(self,
    student_path: str = "data/output/search_results/dataset_docs_public.json",
    right_answers_path: str = "data/datasets/AnsweredQuestions/dataset_docs_public.json",
    k: int = 5):
        print(
            "Evaluation Results\n"
            "========================================\n"
            )
        evaluate(student_path, right_answers_path, 1)
        evaluate(student_path, right_answers_path, 3)
        evaluate(student_path, right_answers_path, 5)
        evaluate(student_path, right_answers_path, 10)
        evaluate(student_path, right_answers_path, k)


    @staticmethod
    def get_context_texts(sources: list[MinimalSource]) -> list[str]:
        texts: list[str] = []
        for source in sources:
            with open(source.file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            texts.append(content[source.first_character_index:source.last_character_index])
        return texts


    def answer(self, query:str = ""):

        context: list[MinimalSearchResults] = self.search(query=query)

        contex_strs: list[str] = self.get_context_texts(context)

        answer = answerer(query, contex_strs)

        return answer

    def answer_dataset(self,
        student_search_results_path: str = "data/output/search_results/dataset_docs_public.json",
        save_directory: str = "data/output/search_results_and_answer"
    ) -> list[AnsweredQuestion]:

            final_result: list[AnsweredQuestion] = []

            with open(student_search_results_path) as f:
                questions: StudentSearchResults = json.load(f)

            for q in tqdm.tqdm(questions["search_results"],  desc="answering questions"):

                final_result.append(
                    AnsweredQuestion(
                        sources=q["retrieved_sources"],
                        answer=self.answer(q["question"]),
                        question_id=q["question_id"],
                        question=q["question"]
                    )
                )

            pathlib.Path(save_directory).mkdir(parents=True, exist_ok=True)

            with open(f"{save_directory}/dataset_docs_public.json", "w") as f:
                json.dump([res.model_dump() for res in final_result], f, indent=4)

            print(f"Loaded {len(final_result)} questions from {student_search_results_path}")
            print(f"Processed {len(final_result)} of {len(final_result)} questions")
            print(f"Saved student_search_results_and_answer to {save_directory}/dataset_docs_public.json")


# class UnansweredQuestion(BaseModel):
#     question_id: str = Field(default_factory=lambda:
#     str(uuid.uuid4()))
#     question: str


# class AnsweredQuestion(UnansweredQuestion):
#     sources: List[MinimalSource]
#     answer: str


# uv run python -m student answer_dataset

# --student_search_results_path data/output/search_results/dataset_docs_public.json
# --save_directory data/output/search_results_and_answer
# Loaded 100 questions from data/output/search_results/dataset_docs_public.json
# Processed 100 of 100 questions
# Saved student_search_results_and_answer to data/output/search_results_and_answer/dataset_docs_public.json


fire.Fire(RAG)
