import numpy as np


MODEL_ID = "intfloat/multilingual-e5-large"
MODEL_REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
EMBEDDING_SIZE = 1024


class E5Encoder:
    def __init__(self, device: str = "auto") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна в текущей установке PyTorch.")

        self.device = device
        self._torch = torch
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self._model = AutoModel.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=dtype, use_safetensors=True
        ).to(device).eval()

    def _encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        tokens = self._tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            hidden = self._model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = hidden.masked_fill(mask == 0, 0).sum(dim=1, dtype=torch.float32)
            pooled = pooled / mask.sum(dim=1)
            return torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {text}" for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}"])[0]

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"query: {text}" for text in texts])
