from threading import Lock
from typing import Any


_model: Any | None = None
_model_name: str | None = None
_model_lock = Lock()


def _load_sentence_transformer(model_name: str) -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(1)
    return SentenceTransformer(model_name)


def get_embedding_model(model_name: str) -> Any:
    global _model, _model_name

    with _model_lock:
        if _model is None:
            _model = _load_sentence_transformer(model_name)
            _model_name = model_name
        elif _model_name != model_name:
            raise RuntimeError(
                f"Embedding model '{_model_name}' is already loaded; "
                f"cannot also load '{model_name}' in this process"
            )
        return _model