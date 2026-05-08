"""Build / load a FAISS IndexFlatIP over the triple embeddings."""

from __future__ import annotations

import os

import faiss
import numpy as np

from ..config import DATA_PATH


def build_or_load_faiss_index(
    triple_embeddings: np.ndarray,
    data_path: str = DATA_PATH,
    file_name: str = "faiss_triples.index",
):
    """Cosine-similarity FAISS index (IndexFlatIP on L2-normalized vectors)."""
    faiss_index_file = os.path.join(data_path, file_name)
    if os.path.exists(faiss_index_file):
        return faiss.read_index(faiss_index_file)

    triple_embeddings = np.array(triple_embeddings).astype("float32")
    faiss.normalize_L2(triple_embeddings)

    dim = triple_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(triple_embeddings)
    print("Indexed triples:", index.ntotal)

    faiss.write_index(index, faiss_index_file)
    return index
