"""Handles answer generation using the LLM."""
from .llm_model import Small_LLM_Model
from functools import lru_cache


@lru_cache(maxsize=1)
def get_llm() -> Small_LLM_Model:
    """Lazy loads the LLM model to save memory and startup time."""
    return Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")


def answerer(query: str, context: list[str], token_limit: int = 150) -> str:
    """Generates an answer using the LLM."""
    llm = get_llm()
    context_str = "\n\n".join(context)
    augmented_prompt = f"""
    Context:
    {context_str}

    Question:
    {query}

    Answer:
    """
    return str(llm.generate(augmented_prompt, max_tokens=token_limit))