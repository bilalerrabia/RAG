import json

def evaluate(student_path: str, right_answers_path: str, k: int) -> float:

    with open(student_path, encoding='utf-8') as f:
        student_answers = json.load(f)

    with open(right_answers_path, encoding='utf-8') as f:
        right_answers = json.load(f)

    gt_map = { q["question_id"]: q["sources"] for q in right_answers["rag_questions"]}

    total_score = 0
    total_questions = 0

    for result in student_answers["search_results"]:

        q_id             = result["question_id"]
        retrieved        = result["retrieved_sources"][:k]
        correct_sources  = gt_map.get(q_id, [])

        if not correct_sources:
            continue

        found = 0

        for correct_source in correct_sources:
            for answer in retrieved:

                if answer["file_path"] != correct_source["file_path"]:
                    continue

                overlap_start  = max(answer["first_character_index"], correct_source["first_character_index"])
                overlap_end    = min(answer["last_character_index"],  correct_source["last_character_index"])
                overlap_length = overlap_end - overlap_start

                if overlap_length <= 0:
                    continue

                correct_length = correct_source["last_character_index"] - correct_source["first_character_index"]
                overlap_ratio  = overlap_length / correct_length

                if overlap_ratio >= 0.05:
                    found += 1
                    break

        question_score = found / len(correct_sources)
        total_score   += question_score
        total_questions += 1

    if total_questions == 0:
        print(f"Recall@{k} = 0 (no valid questions)")
        return 0.0
    recall: float = total_score / total_questions

    print(f"recall@{k} score = {recall}")

    return recall