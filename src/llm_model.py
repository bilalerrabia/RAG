from transformers import AutoModelForCausalLM, AutoTokenizer


class Small_LLM_Model:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        messages = [
            {"role": "system", "content": (
                "You are a code assistant."
                " Answer concisely using only the context.")},
            {"role": "user", "content": prompt}
        ]
        # add special caracteres <|im_start|>, <|im_end|> ...
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # <|im_start|>assistant\n
            enable_thinking=False  # <think>...</think>
        )

        # PyTorch
        inputs = self.tokenizer(text, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id  # End of Sequence => stop
        )

        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

        return str(self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
            ))