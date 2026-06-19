import fire
import chromadb
import tqdm
import json
import pathlib
from .chunker import loader
from .models import MinimalSearchResults, StudentSearchResults, AnsweredQuestion, MinimalSource, StudentSearchResultsAndAnswer, MinimalAnswer
from .indexer import indexer
from .retriever import hybrid_search
from .evaluator import evaluate
from .answerer import answerer


import chromadb
from functools import lru_cache

@lru_cache(maxsize=1)
def get_chroma_collection():
    client = chromadb.PersistentClient(path="data/processed")
    return client.get_collection(name="vllm_chunks")

# index          → index the repository                            <- done
# search         → search for a single query                       <- done
# search_dataset → process multiple questions from JSON            <- done
# answer         → answer a single question with context           <- done
# answer_dataset → generate answers from search results            <- done
# evaluate       → evaluate search results against ground truth    <- done

# index -> search_dataset(k) -> evaluate -> answer_dataset

class RAG:


    def index(self, repo_path: str = "data/raw/vllm-0.10.1", repo_to_save: str = "data/processed", max_chunk_size: int = 2000) -> None:

        indexer(repo_path, repo_to_save, max_chunk_size)
        print(f"Ingestion complete! Indices saved under {repo_to_save}")


    def search(self, query: str, chunks_path: str = "data/processed/chunks.json", index_path: str = "data/processed", k: int = 10) -> list[MinimalSource]:
        
        # Use the cached collection!
        collection = get_chroma_collection()

        results: list[MinimalSource] = hybrid_search(
            chunks_path=chunks_path,
            index_path=index_path,
            query=query,
            k=k,
            bm25_weight=0.8,
            chroma_weight=0.2,
            collection=collection  # Pass it in
        )
        return results


    def search_dataset(
            self, dataset_path: str="data/datasets/UnansweredQuestions/dataset_docs_public.json",
            k:int = 10, save_directory: str = "data/output/search_results") -> None:

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
    k: int = 5) -> None:
        print(
            "Evaluation Results\n"
            "========================================\n"
            )
        evaluate(student_path, right_answers_path, 1)
        evaluate(student_path, right_answers_path, 3)
        evaluate(student_path, right_answers_path, 5)
        evaluate(student_path, right_answers_path, 10)
        evaluate(student_path, right_answers_path, k)


    def answer(self, query:str = "", k: int = 10) -> str:

        context: list[MinimalSearchResults] = self.search(query=query, k=k)

        contex_strs: list[str] = self.get_context_texts(context)

        answer = answerer(query, contex_strs)

        return f"\n{answer}"

    @staticmethod
    def get_context_texts(sources: list[MinimalSource]) -> list[str]:
        texts: list[str] = []
        # ONLY pass the top 3 files to the LLM to save massive time
        for source in sources[:3]:
            try:
                with open(source.file_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # ONLY pass the first 1500 characters to the LLM
                texts.append(content[:1500])
            except Exception:
                texts.append("")
        return texts

    def answer_dataset(self,
        student_search_results_path: str = "data/output/search_results/dataset_docs_public.json",
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:

            results: StudentSearchResultsAndAnswer = StudentSearchResultsAndAnswer(
                search_results=[],
                k=0
            )

            with open(student_search_results_path, encoding="utf-8") as f:
                questions = json.load(f)

            results.k = questions["k"]

            for q in tqdm.tqdm(questions["search_results"],  desc="answering questions"):
                # 1. Reuse the sources directly from JSON!
                sources = [MinimalSource(**s) for s in q["retrieved_sources"]]
                
                # 2. Get truncated context (only top 3 files, 1500 chars each)
                contex_strs = self.get_context_texts(sources)
                
                # 3. Call answerer directly (NO self.answer!)
                ans = answerer(q["question"], contex_strs)
                
                results.search_results.append(
                    MinimalAnswer(
                        retrieved_sources=q["retrieved_sources"],
                        answer=ans,
                        question_id=q["question_id"],
                        question=q["question"]
                    )
                )

            pathlib.Path(save_directory).mkdir(parents=True, exist_ok=True)
            filename = pathlib.Path(student_search_results_path).name

            with open(f"{save_directory}/{filename}", "w") as f:
                json.dump(results.model_dump(), f, indent=4)

            print(f"Loaded {len(results.search_results)} questions from {student_search_results_path}")
            print(f"Processed {len(results.search_results)} of {len(results.search_results)} questions")
            print(f"Saved student_search_results_and_answer to {save_directory}/{filename}")

# try:
fire.Fire(RAG)
# except Exception as e:
#     print(f"error: {e}")