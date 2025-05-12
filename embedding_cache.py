import functools
from langchain.embeddings import HuggingFaceEmbeddings

@functools.lru_cache(maxsize=1)
def get_embedder(model_name: str, device: str, normalize: bool = True):
    """
    Lazily loads the Hugging‑Face embedder once per Python process
    and re‑uses it for all subsequent calls.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
    )
