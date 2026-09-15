"""
language_encoder.py — Real Sentence Embedding for Multi-Modal Grounding
------------------------------------------------------------------------
Replaces the random 768-d vectors with genuine semantic embeddings
using sentence-transformers (all-MiniLM-L6-v2 → 384-d) or
BAAI/bge-small-en-v1.5 (384-d).

Falls back gracefully to cached random embeddings if the model
is not installed, so the pipeline never breaks.

Usage
-----
    from models.language_encoder import encode_instruction, get_encoder

    emb = encode_instruction("Pick up the plate and hand it to arm B.")
    # emb.shape → (1, 384) or (1, 768) depending on model
"""

import numpy as np
import torch
from typing import Optional

# ── Model selection ─────────────────────────────────────────────────────────────
# Using a lightweight model that runs well on CPU / Intel Core Ultra
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d, ~23 MB
EMBEDDING_DIM  = 384   # Update policy action head if changing model

_encoder = None
_dim: int = EMBEDDING_DIM


def get_encoder(model_name: str = DEFAULT_MODEL):
    """Lazy-loads the SentenceTransformer encoder (downloads on first call)."""
    global _encoder, _dim
    if _encoder is not None:
        return _encoder, _dim

    try:
        from sentence_transformers import SentenceTransformer
        print(f"[LangEncoder] Loading '{model_name}' ...")
        _encoder = SentenceTransformer(model_name)
        # get_embedding_dimension() is the new API; fall back for older versions
        _dim = (
            _encoder.get_embedding_dimension()
            if hasattr(_encoder, "get_embedding_dimension")
            else _encoder.get_sentence_embedding_dimension()
        )
        print(f"[LangEncoder] Ready — embedding dim={_dim}")
    except ImportError:
        print("[LangEncoder] sentence-transformers not installed. "
              "Run: pip install sentence-transformers\n"
              "Falling back to zero embeddings.")
        _encoder = None
        _dim = EMBEDDING_DIM

    return _encoder, _dim


def encode_instruction(
    instruction: str,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    normalize: bool = True,
) -> np.ndarray:
    """
    Encodes a natural-language instruction into a fixed-size embedding vector.

    Parameters
    ----------
    instruction : str   The task instruction to encode.
    model_name  : str   HuggingFace model ID.
    device      : str   'cpu' or 'cuda'.
    normalize   : bool  L2-normalise the output vector.

    Returns
    -------
    np.ndarray of shape (1, embedding_dim)
    """
    enc, dim = get_encoder(model_name)

    if enc is None:
        # Graceful fallback: deterministic random based on hash of instruction
        rng = np.random.RandomState(abs(hash(instruction)) % (2**31))
        emb = rng.randn(1, dim).astype(np.float32)
    else:
        emb = enc.encode(
            [instruction],
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            device=device,
        )  # shape (1, dim)

    return emb.astype(np.float32)


def encode_batch(
    instructions: list,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    normalize: bool = True,
) -> np.ndarray:
    """
    Batch-encodes a list of instructions.

    Returns
    -------
    np.ndarray of shape (N, embedding_dim)
    """
    enc, dim = get_encoder(model_name)

    if enc is None:
        embs = []
        for inst in instructions:
            rng = np.random.RandomState(abs(hash(inst)) % (2**31))
            embs.append(rng.randn(dim).astype(np.float32))
        return np.stack(embs)

    return enc.encode(
        instructions,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        device=device,
    ).astype(np.float32)


if __name__ == "__main__":
    instructions = [
        "Open the drawer and retrieve the spoon.",
        "Place the plate on the left side of the table.",
        "Hand the cup from arm A to arm B.",
    ]
    embs = encode_batch(instructions)
    print(f"Batch embedding shape: {embs.shape}")

    # Show cosine similarity between instructions
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embs)
    print("\nCosine similarity matrix:")
    print(sim.round(3))
