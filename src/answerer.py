from .llm_model import Small_LLM_Model
from functools import lru_cache

# Lazy load the model so it doesn't load during indexing/searching
@lru_cache(maxsize=1)
def get_llm():
    return Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")

def answerer(query: str, context: list[str], token_limit: int = 150) -> str:
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