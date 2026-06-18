import fire
import tqdm
import json
import pathlib
from .chunker import loader
from .models import MinimalSearchResults, StudentSearchResults, AnsweredQuestion, MinimalSource
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

# index -> search_dataset(k) -> evaluate -> answer_dataset

class RAG:

    def index(self, repo_path: str = "data/raw/vllm-0.10.1", repo_to_save: str = "data/processed", max_chunk_size: int = 2000) -> None:

        indexer(repo_path, repo_to_save, max_chunk_size)
        print(f"Ingestion complete! Indices saved under {repo_to_save}")


    def search(self, query: str,
    chunks_path: str = "data/processed/chunks.json", index_path: str = "data/processed",
    k: int = 10) -> list[MinimalSource]:

        results: list[MinimalSource] = retrieval(
            chunks_path=chunks_path,
            index_path=index_path,
            query=query,
            k=k
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


    @staticmethod
    def get_context_texts(sources: list[MinimalSource]) -> list[str]:
        texts: list[str] = []
        for source in sources:
            with open(source.file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            texts.append(content[source.first_character_index:source.last_character_index])
        return texts


    def answer(self, query:str = "", k: int = 10) -> str:

        context: list[MinimalSearchResults] = self.search(query=query, k=k)

        contex_strs: list[str] = self.get_context_texts(context)

        answer = answerer(query, contex_strs)

        return f"\n{answer}"

# add StudentSearchResultsAndAnswer to answer_dataset

    def answer_dataset(self,
        student_search_results_path: str = "data/output/search_results/dataset_docs_public.json",
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:

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

try:
    fire.Fire(RAG)
except Exception as e:
    print(f"error: {e}")