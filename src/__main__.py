"""Command-Line Interface for the RAG pipeline."""
import fire
import chromadb
import tqdm
import json
import pathlib
from functools import lru_cache
from .models import MinimalSearchResults, StudentSearchResults, MinimalSource, StudentSearchResultsAndAnswer, MinimalAnswer
from .indexer import indexer
from .retriever import hybrid_search
from .evaluator import evaluate
from .answerer import answerer
from typing import Any


@lru_cache(maxsize=1)
def get_chroma_collection() -> Any:
    """Caches ChromaDB collection to avoid re-initializing on every query."""
    client = chromadb.PersistentClient(path="data/processed")
    return client.get_collection(name="vllm_chunks")


class RAG:
    """RAG CLI commands."""
    
    def index(self, repo_path: str = "data/raw/vllm-0.10.1", repo_to_save: str = "data/processed", max_chunk_size: int = 2000) -> None:
        """Index the repository."""
        try:
            indexer(repo_path, repo_to_save, max_chunk_size)
            print(f"Ingestion complete! Indices saved under {repo_to_save}")
        except Exception as e:
            print(f"Error during indexing: {e}")

    def search(self, query: str, chunks_path: str = "data/processed/chunks.json", index_path: str = "data/processed", k: int = 10) -> list[MinimalSource]:
        """Search for a single query."""
        collection = get_chroma_collection()
        return hybrid_search(
            chunks_path=chunks_path,
            index_path=index_path,
            query=query,
            k=k,
            bm25_weight=0.6,
            chroma_weight=0.4,
            collection=collection
        )

    def search_dataset(self, dataset_path: str = "data/datasets/UnansweredQuestions/dataset_docs_public.json", k: int = 10, save_directory: str = "data/output/search_results") -> None:
        """Process multiple questions and output search results."""
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
        res = StudentSearchResults(k=k, search_results=answers)
        with open(f'{save_directory}/{filename}', "w", encoding="utf-8") as f:
            json.dump(res.model_dump(), f, indent=4)
        print(f"Saved student_search_results to {save_directory}/{filename}")

    def evaluate(self, student_path: str = "data/output/search_results/dataset_docs_public.json", right_answers_path: str = "data/datasets/AnsweredQuestions/dataset_docs_public.json", k: int = 5) -> None:
        """Evaluate search results against ground truth."""
        print("Evaluation Results\n========================================\n")
        evaluate(student_path, right_answers_path, 1)
        evaluate(student_path, right_answers_path, 3)
        evaluate(student_path, right_answers_path, 5)
        evaluate(student_path, right_answers_path, 10)
        evaluate(student_path, right_answers_path, k)

    @staticmethod
    def get_context_texts(sources: list[MinimalSource]) -> list[str]:
        """Get context texts from sources."""
        texts: list[str] = []
        for source in sources[:3]:
            try:
                with open(source.file_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                texts.append(content[:1500])
            except Exception:
                texts.append("")
        return texts

    def answer(self, query: str = "", k: int = 10) -> str:
        """Answer a single question with context."""
        context = self.search(query=query, k=k)
        contex_strs = self.get_context_texts(context)
        return answerer(query, contex_strs)

    def answer_dataset(self, student_search_results_path: str = "data/output/search_results/dataset_docs_public.json", save_directory: str = "data/output/search_results_and_answer") -> None:
        """Generate answers from search results."""
        results = StudentSearchResultsAndAnswer(search_results=[], k=0)
        with open(student_search_results_path, encoding="utf-8") as f:
            questions = json.load(f)
        results.k = questions["k"]

        for q in tqdm.tqdm(questions["search_results"], desc="answering questions"):
            sources = [MinimalSource(**s) for s in q["retrieved_sources"]]
            context_strs = self.get_context_texts(sources)
            q_text = q.get("question_str") or q.get("question")
            ans = answerer(q_text, context_strs)
            results.search_results.append(
                MinimalAnswer(
                    retrieved_sources=q["retrieved_sources"],
                    answer=ans,
                    question_id=q["question_id"],
                    question_str=q_text
                )
            )

        pathlib.Path(save_directory).mkdir(parents=True, exist_ok=True)
        filename = pathlib.Path(student_search_results_path).name
        with open(f"{save_directory}/{filename}", "w", encoding="utf-8") as f:
            json.dump(results.model_dump(), f, indent=4)
        print(f"Loaded {len(results.search_results)} questions from {student_search_results_path}")
        print(f"Processed {len(results.search_results)} of {len(results.search_results)} questions")
        print(f"Saved student_search_results_and_answer to {save_directory}/{filename}")

    def pipeline(self):
        self.index()
        self.search_dataset()
        self.evaluate()

def main():
    fire.Fire(RAG)


if __name__ == "__main__":
    main()