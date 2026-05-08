"""Baseline dense retriever (cosine similarity over MiniLM embeddings)."""

from __future__ import annotations

import os
import pickle
from typing import List, Tuple

import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sklearn.metrics.pairwise import cosine_similarity

from .config import EMBED_MODEL_NAME
from .data import get_factID_from_factLabel


def compute_document_embeddings_optimized(
    documents: List[str], embedding_model, batch_size: int = 16
) -> np.ndarray:
    """Compute embeddings for all documents in batches."""
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batch_embeddings = embedding_model.get_text_embedding_batch(batch)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)


class CustomRetriever:
    """Embed a corpus of documents and retrieve top-k by cosine similarity."""

    def __init__(
        self,
        documents: List[str],
        embed_model_name: str = EMBED_MODEL_NAME,
        cache_file: str = "embeddings_cache.pkl",
        batch_size: int = 32,
        n_workers: int = 4,
    ):
        self.documents = documents
        self.embed_model_name = embed_model_name
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.n_workers = n_workers

        self.embed_model = HuggingFaceEmbedding(
            model_name=self.embed_model_name, trust_remote_code=True
        )

        if self._load_embeddings_from_cache():
            print(f"\tEmbeddings loaded from cache: {self.cache_file}")
        else:
            print(
                f"\tComputing embeddings for {len(documents)} documents "
                f"with {self.embed_model_name}..."
            )
            self.document_embeddings = self._compute_document_embeddings_optimized()
            self._save_embeddings_to_cache()
            print(f"\tEmbeddings cached to: {self.cache_file}")

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------
    def _load_embeddings_from_cache(self) -> bool:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                if (
                    cache_data["documents"] == self.documents
                    and cache_data["model_name"] == self.embed_model_name
                ):
                    self.document_embeddings = cache_data["embeddings"]
                    return True
            except Exception:
                print("\tCache load failed, recomputing embeddings...")
        return False

    def _save_embeddings_to_cache(self) -> None:
        cache_data = {
            "documents": self.documents,
            "model_name": self.embed_model_name,
            "embeddings": self.document_embeddings,
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache_data, f)

    def _compute_document_embeddings_optimized(self) -> np.ndarray:
        return compute_document_embeddings_optimized(
            self.documents, self.embed_model, batch_size=self.batch_size
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3):
        if not hasattr(self, "_query_cache"):
            self._query_cache = {}

        if query in self._query_cache:
            query_embedding = self._query_cache[query]
        else:
            query_embedding = self.embed_model.get_text_embedding(query)
            query_embedding = np.array(query_embedding).reshape(1, -1)
            self._query_cache[query] = query_embedding

        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved_docs = []
        for idx in top_indices:
            doc_obj = type("Document", (), {"text": self.documents[idx]})()
            retrieved_docs.append(doc_obj)
        return retrieved_docs


def batch_retrieve(
    questions: List[str],
    retriever: CustomRetriever,
    facts_dataframe,
    batch_size: int = 64,
    top_k: int = 3,
) -> Tuple[List[List[int]], List[List[str]]]:
    """Vectorized retrieval over many questions at once."""
    print(f"Computing embeddings for {len(questions)} queries in batches...")
    embeddings = []
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i : i + batch_size]
        batch_embeddings = retriever.embed_model.get_text_embedding_batch(batch_questions)
        embeddings.extend(batch_embeddings)

    query_embeddings = np.array(embeddings)
    similarities = cosine_similarity(query_embeddings, retriever.document_embeddings)

    retrieved_facts_ids: List[List[int]] = []
    retrieved_facts_labels: List[List[str]] = []
    for question_similarities in similarities:
        top_indices = np.argsort(question_similarities)[::-1][:top_k]
        retrieved_labels = [retriever.documents[idx] for idx in top_indices]
        retrieved_ids = [
            get_factID_from_factLabel(label, facts_dataframe)
            for label in retrieved_labels
        ]
        retrieved_facts_ids.append(retrieved_ids)
        retrieved_facts_labels.append(retrieved_labels)
    return retrieved_facts_ids, retrieved_facts_labels
