#!/usr/bin/env python3
"""Build a Kaggle submission CSV for the test split using QUASAR.

Run:
    python scripts/make_submission.py --top-k 5 --output submission.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    sentence_model = SentenceTransformer(EMBED_MODEL_NAME, device=args.device)

    df = load_triples_dataframe(args.data_path)
    triples = build_facts_plus(df, args.data_path)
    _, _, rel2emb = build_relation_embeddings(triples, sentence_model, args.data_path)
    nlentity2emb = build_nl_entity_embeddings(triples, sentence_model, args.data_path)
    entity2emb = build_qid_embeddings(triples, sentence_model, args.data_path)
    triple_embeddings = build_triple_embeddings(
        triples, sentence_model, rel2emb, args.data_path
    )
    faiss_index = build_or_load_faiss_index(triple_embeddings, args.data_path)

    model, tokenizer = load_model_and_tokenizer()
    pipe = build_pipeline(model, tokenizer)
    pm = PromptManager()
    dataset = load_dataset(args.data_path)

    run_quasar_evaluation(
        dataset=dataset,
        sentence_model=sentence_model,
        faiss_index=faiss_index,
        triples=triples,
        entity2emb=entity2emb,
        nlentity2emb=nlentity2emb,
        pm=pm,
        pipe=pipe,
        top_k=args.top_k,
        prompting_config={"prompt_type": "standard_id"},
        data_path=args.data_path,
        split="test",
    )

    # Re-load the response cache just written and produce the submission file.
    response_cache = os.path.join(
        args.data_path,
        "cache",
        f"test_data_model_response_retrieved_top{args.top_k}_id_V2.csv",
    )
    df_resp = pd.read_csv(response_cache)
    submission = df_resp[["id", "model_response"]].rename(
        columns={"model_response": "answer"}
    )
    submission.to_csv(args.output, index=False)
    print(f"Submission saved to {args.output}")


if __name__ == "__main__":
    main()
