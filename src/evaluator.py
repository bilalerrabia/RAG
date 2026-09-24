import json
from typing import Any


def is_found(retrieved: Any, correct: Any) -> Any:
    """Checks if a retrieved source overlaps a
    correct source by at least 5%."""
    if retrieved["file_path"] != correct["file_path"]:
        return False

    overlap_start = max(
        retrieved["first_character_index"],
        correct["first_character_index"])
    overlap_end = min(
        retrieved["last_character_index"],
        correct["last_character_index"])
    overlap_length = overlap_end - overlap_start

    if overlap_length <= 0:
        return False

    correct_length = (
        correct["last_character_index"] - correct["first_character_index"])
    return (overlap_length / correct_length) >= 0.05


def evaluate(student_path: str, right_answers_path: str, k: int) -> float:
    """Evaluates student answers against right answers using Recall@k."""

    with open(student_path) as f:
        student_answers = json.load(f)

    with open(right_answers_path) as f:
        right_answers = json.load(f)

    ra_map = {
        q["question_id"]: q["sources"] for q in
        right_answers["rag_questions"]}

    total_score = 0.0
    questions_counter = 0

    for result in student_answers["search_results"]:
        correct_sources = ra_map.get(result["question_id"])
        if not correct_sources:
            continue

        retrieved = result["retrieved_sources"][:k]

        found = sum(
            1 for c_src in correct_sources
            if any(is_found(r_src, c_src) for r_src in retrieved)
        )

        total_score += found / len(correct_sources)
        questions_counter += 1

    recall = total_score / questions_counter if questions_counter > 0 else 0.0
    print(f"recall@{k} score = {recall}")
    return recall
