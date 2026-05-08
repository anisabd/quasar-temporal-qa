#!/usr/bin/env python3
"""Run the QUASAR retrieval + LLM evaluation pipeline.

Builds (or reuses) FAISS resources, then evaluates QUASAR on the chosen split.

Run:
    python scripts/run_quasar.py --top-k 5 --split val
"""
from __future__ import annotations

import argparse
import os
import sys

from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quasar_rag.config import DATA_PATH, EMBED_MODEL_NAME
from quasar_rag.data import load_dataset
from quasar_rag.llm import build_pipeline, load_model_and_tokenizer
from quasar_rag.prompting import PromptManager
from quasar_rag.quasar.embeddings import (
    build_nl_entity_embeddings,
    build_qid_embeddings,
    build_relation_embeddings,
    build_triple_embeddings,
)
from quasar_rag.quasar.index import build_or_load_faiss_index
from quasar_rag.quasar.pipeline import run_quasar_evaluation
from quasar_rag.quasar.triples import build_facts_plus, load_triples_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embed-model", default=EMBED_MODEL_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prompt-type",
        default="standard_id",
        choices=["standard_nl", "standard_id"],
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Sentence model + triples + embeddings
    # ------------------------------------------------------------------
    print(f"Loading sentence model {args.embed_model} on {args.device}...")
    sentence_model = SentenceTransformer(args.embed_model, device=args.device)

    df = load_triples_dataframe(args.data_path)
    triples = build_facts_plus(df, args.data_path)
    _, _, rel2emb = build_relation_embeddings(triples, sentence_model, args.data_path)
    nlentity2emb = build_nl_entity_embeddings(triples, sentence_model, args.data_path)
    entity2emb = build_qid_embeddings(triples, sentence_model, args.data_path)
    triple_embeddings = build_triple_embeddings(
        triples, sentence_model, rel2emb, args.data_path
    )
    faiss_index = build_or_load_faiss_index(triple_embeddings, args.data_path)

    # ------------------------------------------------------------------
    # 2. LLM
    # ------------------------------------------------------------------
    print("Loading LLM...")
    model, tokenizer = load_model_and_tokenizer()
    pipe = build_pipeline(model, tokenizer)
    pm = PromptManager()

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    dataset = load_dataset(args.data_path)
    results = run_quasar_evaluation(
        dataset=dataset,
        sentence_model=sentence_model,
        faiss_index=faiss_index,
        triples=triples,
        entity2emb=entity2emb,
        nlentity2emb=nlentity2emb,
        pm=pm,
        pipe=pipe,
        top_k=args.top_k,
        prompting_config={"prompt_type": args.prompt_type},
        data_path=args.data_path,
        split=args.split,
    )
    print("\nFinal results:")
    for k, v in results.items():
        print(f"  top_k={k}: exact_F1={v['exact_f1']:.4f}  partial_F1={v['partial_f1']:.4f}")


if __name__ == "__main__":
    main()
