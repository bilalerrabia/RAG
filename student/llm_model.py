import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, logging
from huggingface_hub import hf_hub_download


class Small_LLM_Model:

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:

        self._model_name = model_name
        self._device = "cpu"

        if dtype is None:
            dtype = torch.float32
        self._dtype = dtype

        # --- load tokenizer & model -------------------------------------------------
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

        # switch to inference-only mode
        for p in self._model.parameters():
            p.requires_grad = False


    def encode(self, text: str) -> torch.Tensor:
        """Tokenise *text* and return a 2-D ``input_ids`` tensor on the target device."""
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)


    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Inverse of :py:meth:`encode`. Removes special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True)


    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """
        Given a list of input token ids, return the raw logits (no softmax) for the next token.
        """
        input_tensor = torch.tensor([input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            out = self._model(input_ids=input_tensor)
        # Get logits for the last token in the sequence for the batch (batch size 1)
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits]


    def get_path_to_vocab_file(self) -> str:
        vocab_file_name = self._tokenizer.vocab_files_names.get('vocab_file', "vocab.json")
        vocab_path = hf_hub_download(
            repo_id=self._model_name,
            filename=vocab_file_name
        )
        return vocab_path


    def get_path_to_merges_file(self) -> str:
        merges_file_name = self._tokenizer.vocab_files_names.get('merges_file', "merges.txt")
        merges_path = hf_hub_download(
            repo_id=self._model_name,
            filename=merges_file_name
        )
        return merges_path


    def get_path_to_tokenizer_file(self) -> str:
        tokenizer_file_name = self._tokenizer.vocab_files_names.get('tokenizer_file', "tokenizer.json")
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name,
            filename=tokenizer_file_name
        )
        return tokenizer_path
