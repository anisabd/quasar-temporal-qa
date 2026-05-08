#!/usr/bin/env python3
"""Baseline RAG (Section 4): MiniLM retriever over flat facts + Llama-3.2-3B.

Run:
    python scripts/run_rag.py --top-k 1 3 5 10 --prompt-type standard_id
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quasar_rag.config import CACHE_DIR, DATA_PATH
from quasar_rag.data import load_dataset, load_facts_dataframe
from quasar_rag.llm import build_pipeline, load_model_and_tokenizer
from quasar_rag.prompting import PromptManager
from quasar_rag.rag_pipeline import (
    display_comparison_results,
    plot_comparison_results,
    run_rag_evaluation,
)
from quasar_rag.retriever import CustomRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--prompt-type",
        default="standard_id",
        choices=["standard_nl", "standard_id"],
    )
    parser.add_argument("--num-shots", type=int, default=10)
    parser.add_argument("--save-plot", default="docs/figures/rag_topk_comparison.png")
    args = parser.parse_args()

    dataset = load_dataset(args.data_path)
    facts_dataframe = load_facts_dataframe(args.data_path)

    print("Initializing baseline retriever (MiniLM)...")
    retriever = CustomRetriever(
        facts_dataframe["fact"].to_list(),
        cache_file=os.path.join(CACHE_DIR, "embeddings_cache.pkl"),
    )

    print("Loading LLM...")
    model, tokenizer = load_model_and_tokenizer()
    pipe = build_pipeline(model, tokenizer)
    pm = PromptManager()

    results = run_rag_evaluation(
        dataset=dataset,
        retriever=retriever,
        facts_dataframe=facts_dataframe,
        pm=pm,
        pipe=pipe,
        top_k_values=args.top_k,
        prompting_config={
            "prompt_type": args.prompt_type,
            "num_shots": args.num_shots,
        },
        data_path=args.data_path,
        split=args.split,
    )

    display_comparison_results(results)
    if args.save_plot:
        os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
        plot_comparison_results(results, save_path=args.save_plot)


if __name__ == "__main__":
    main()
