from .llm_model import Small_LLM_Model

def answerer(query: str, context: list[str], token_limits: int = 200, model_name: str = "Qwen/Qwen3-0.6B") -> str:

    augmented_prompt: str = f"""You are a helpful and truthful AI assistant. 
        Use the following pieces of retrieved context to answer the user's question completely and accurately. 
        If you do not know the answer based strictly on the context, state that you do not know, and do not make up information.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

    llm_model = Small_LLM_Model(model_name=model_name)

    res = llm_model.generate(
        augmented_prompt,
        max_tokens=token_limits
        )

    return res

