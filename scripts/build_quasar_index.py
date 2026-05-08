#!/usr/bin/env python3
"""Build the QUASAR resources: triple/relation/entity embeddings + FAISS index.

Run:
    python scripts/build_quasar_index.py
"""
from __future__ import annotations

import argparse
import os
import sys

from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quasar_rag.config import DATA_PATH, EMBED_MODEL_NAME
from quasar_rag.quasar.embeddings import (
    build_nl_entity_embeddings,
    build_qid_embeddings,
    build_relation_embeddings,
    build_triple_embeddings,
)
from quasar_rag.quasar.index import build_or_load_faiss_index
from quasar_rag.quasar.triples import build_facts_plus, load_triples_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embed-model", default=EMBED_MODEL_NAME)
    args = parser.parse_args()

    print(f"Loading sentence model {args.embed_model} on {args.device}...")
    model = SentenceTransformer(args.embed_model, device=args.device)

    print("Loading triples dataframe...")
    df = load_triples_dataframe(args.data_path)
    print(f"  {len(df)} valid triples")

    print("Enriching triples with QIDs...")
    triples = build_facts_plus(df, args.data_path)

    print("Building relation / NL-entity / QID embeddings...")
    _, _, rel2emb = build_relation_embeddings(triples, model, args.data_path)
    nlentity2emb = build_nl_entity_embeddings(triples, model, args.data_path)
    entity2emb = build_qid_embeddings(triples, model, args.data_path)
    print(
        f"  rel2emb={len(rel2emb)} | nlentity2emb={len(nlentity2emb)} | "
        f"entity2emb={len(entity2emb)}"
    )

    print("Building triple embeddings...")
    triple_embeddings = build_triple_embeddings(triples, model, rel2emb, args.data_path)

    print("Building FAISS index...")
    index = build_or_load_faiss_index(triple_embeddings, args.data_path)
    print(f"FAISS index ready ({index.ntotal} triples).")


if __name__ == "__main__":
    main()
