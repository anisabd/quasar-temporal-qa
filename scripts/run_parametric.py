#!/usr/bin/env python3
"""Parametric-memory baseline: prompt the LLM with no retrieval.

Generates answers (NL or ID) and evaluates them against the gold split.

Run:
    python scripts/run_parametric.py --prompt-type standard_nl --num-shots 0
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quasar_rag.config import CACHE_DIR, DATA_PATH
from quasar_rag.data import load_dataset
from quasar_rag.evaluation import BaseEvaluator
from quasar_rag.llm import build_pipeline, generate_all_responses_batch, load_model_and_tokenizer
from quasar_rag.prompting import (
    PromptManager,
    create_all_prompts,
    create_response_dataframe,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--prompt-type",
        default="standard_nl",
        choices=["basic", "standard_nl", "standard_id", "standard_nl2"],
    )
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--cache-name", default=None, help="Override the response cache filename."
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data_path)
    pm = PromptManager()

    cache_file = args.cache_name or os.path.join(
        CACHE_DIR,
        f"{args.split}_data_model_response_{args.prompt_type}"
        f"{'_fw' if args.num_shots else ''}.csv",
    )

    if os.path.exists(cache_file):
        print(f"Loading cached responses from {cache_file}")
        df_response = pd.read_csv(cache_file)
    else:
        print("Loading model & pipeline...")
        model, tokenizer = load_model_and_tokenizer()
        pipe = build_pipeline(model, tokenizer)

        print("Building prompts...")
        prompts = create_all_prompts(
            dataset_df=dataset[args.split],
            prompt_manager=pm,
            prompt_type=args.prompt_type,
            num_shots=args.num_shots,
            train_df=dataset["train"] if args.num_shots else None,
        )
        print("Generating responses...")
        responses = generate_all_responses_batch(
            pipe,
            prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        df_response = create_response_dataframe(
            dataset[args.split],
            responses,
            config={
                "prompt_type": args.prompt_type,
                "num_shots": args.num_shots,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "batch_size": args.batch_size,
            },
        )
        df_response.to_csv(cache_file, index=False)
        print(f"Saved responses to {cache_file}")

    truth_col = "original_answers" if args.prompt_type == "standard_id" else "natural_lang_answers"
    print(f"\nEvaluating ({truth_col})")
    evaluator = BaseEvaluator(df_response, truth_col=truth_col, pred_col="model_response")
    evaluator.evaluate_exact_partial(id=(args.prompt_type == "standard_id"))


if __name__ == "__main__":
    main()
