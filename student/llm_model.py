import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Small_LLM_Model:

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate(self, prompt: str, max_tokens: int = 200) -> str:

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

        return str(self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ))
