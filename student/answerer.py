from .llm_model import Small_LLM_Model

llm_model = Small_LLM_Model(
    model_name="Qwen/Qwen3-0.6B"
)

def answerer(query: str, context: list[str], token_limits: int = 200) -> str:

    augmented_prompt = f"""
    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    return llm_model.generate(
        augmented_prompt,
        max_tokens=token_limits
    )