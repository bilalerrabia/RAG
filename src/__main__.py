import fire
import chromadb
import tqdm
import json
import pathlib
from functools import lru_cache
from typing import Any
from .models import MinimalSearchResults, StudentSearchResults
from .models import MinimalSource, StudentSearchResultsAndAnswer, MinimalAnswer
from .indexer import indexer
from .retriever import hybrid_search
from .evaluator import evaluate
from .answerer import answerer


@lru_cache()
def get_chroma_collection() -> Any:
    client = chromadb.PersistentClient(path="data/processed")
    return client.get_collection(name="vllm_chunks")


class RAG:
    def index(
        self, repo_path: str = "data/raw/vllm-0.10.1",
        repo_to_save: str = "data/processed", max_chunk_size: int = 2000
    ) -> None:
        try:
            indexer(repo_path, repo_to_save, max_chunk_size)
            print(f"Ingestion complete! Indices saved under {repo_to_save}")
        except Exception as e:
            print(f"Error during indexing: {e}")

    def search(
            self, query: str,
            chunks_path: str = "data/processed/chunks.json",
            index_path: str = "data/processed",
            k: int = 10
    ) -> Any:
        collection = get_chroma_collection()
        return hybrid_search(
            chunks_path=chunks_path,
            index_path=index_path,
            query=query, k=k,
            collection=collection
        )

    def search_dataset(
        self,
        dataset_path: str = (
            "data/datasets/UnansweredQuestions/"
            "dataset_docs_public.json"
            ),
        k: int = 10,
        save_directory: str = "data/output/search_results"
    ) -> None:
        filename = pathlib.Path(dataset_path).name
        answers: list[MinimalSearchResults] = []

        with open(dataset_path, encoding="utf-8") as f:
            questions = json.load(f)

        for q in tqdm.tqdm(questions["rag_questions"], desc="searching"):
            q_text = q.get("question_str") or q.get("question")
            answers.append(MinimalSearchResults(
                question_id=q["question_id"],
                question_str=q_text,
                retrieved_sources=self.search(query=q_text, k=k)
            ))

        pathlib.Path(save_directory).mkdir(parents=True, exist_ok=True)

        res: StudentSearchResults = StudentSearchResults(
            k=k,
            search_results=answers
        )

        with open(f'{save_directory}/{filename}', "w") as f:
            json.dump(res.model_dump(), f, indent=4)

        print(f"Saved student_search_results to {save_directory}/{filename}")

    def evaluate(
        self,
        student_path: str = (
            "data/output/search_results"
            "/dataset_docs_public.json"
        ),
        right_answers_path: str = (
            "data/datasets/AnsweredQuestions"
            "/dataset_docs_public.json"
            ),
        k: int = 5
    ) -> None:
        print("Evaluation Results\n========================================\n")
        evaluate(student_path, right_answers_path, 1)
        evaluate(student_path, right_answers_path, 3)
        evaluate(student_path, right_answers_path, 5)
        evaluate(student_path, right_answers_path, 10)
        evaluate(student_path, right_answers_path, k)

    @staticmethod
    def get_context_texts(sources: list[MinimalSource], k: int) -> list[str]:
        texts: list[str] = []
        for source in sources[:k]:
            try:
                with open(source.file_path) as f:
                    content = f.read()
                texts.append(
                    content[
                        source.first_character_index:
                        source.last_character_index])
            except Exception:
                texts.append("")
        return texts

    def answer(self, query: str, k: int = 10) -> str:
        sources: list[MinimalSource] = self.search(query=query, k=k)
        context_strs: list[str] = self.get_context_texts(sources, k)
        return str(answerer(query, context_strs))

    def answer_dataset(
        self,
        student_search_results_path: str = (
            "data/output/search_results/"
            "dataset_docs_public.json"
        ),
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        results = StudentSearchResultsAndAnswer(search_results=[], k=0)
        with open(student_search_results_path) as f:
            questions = json.load(f)
        results.k = questions["k"]

        for q in tqdm.tqdm(
                questions["search_results"], desc="answering questions"):
            sources = [MinimalSource(**s) for s in q["retrieved_sources"]]
            q_text = q.get("question_str") or q.get("question")
            ans = answerer(q_text, sources)
            results.search_results.append(
                MinimalAnswer(
                    retrieved_sources=q["retrieved_sources"], answer=ans,
                    question_id=q["question_id"], question_str=q_text
                )
            )

        pathlib.Path(save_directory).mkdir(parents=True, exist_ok=True)
        filename = pathlib.Path(student_search_results_path).name
        with open(f"{save_directory}/{filename}", "w") as f:
            json.dump(results.model_dump(), f, indent=4)
        print(
            f"Saved student_search_results_and_answer "
            f"to {save_directory}/{filename}"
            )

    def pipeline(self) -> None:
        self.index()
        self.search_dataset()
        self.answer_dataset()


if __name__ == "__main__":
    try:
        fire.Fire(RAG)
    except BaseException as e:
        print(e)