#!/usr/bin/env python3
"""Evaluate a fine-tuned (Q-LoRA) model on the validation split.

Run:
    python scripts/eval_finetuned.py \
        --adapter-path data/cache/qlora-natural_lang_answers \
        --prompt-type standard_nl --response-type nl
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quasar_rag.config import CACHE_DIR, DATA_PATH, LLM_MODEL_ID
from quasar_rag.data import load_dataset
from quasar_rag.evaluation import BaseEvaluator
from quasar_rag.finetuning import load_finetuned_model
from quasar_rag.llm import generate_all_responses_batch
from quasar_rag.prompting import (
    PromptManager,
    create_all_prompts,
    create_response_dataframe,
)
from transformers import AutoTokenizer, pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--prompt-type", default="standard_nl", choices=["standard_nl", "standard_id"]
    )
    parser.add_argument("--response-type", default="nl", choices=["nl", "id"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    cache_file = os.path.join(
        CACHE_DIR,
        f"finetuned_{args.response_type}_{args.split}.csv",
    )

    if os.path.exists(cache_file):
        print(f"Loading cached responses from {cache_file}")
        df_response = pd.read_csv(cache_file)
    else:
        print(f"Loading fine-tuned model from {args.adapter_path}...")
        model = load_finetuned_model(LLM_MODEL_ID, args.adapter_path)

        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        pm = PromptManager()
        dataset = load_dataset(args.data_path)

        prompts = create_all_prompts(
            dataset_df=dataset[args.split],
            prompt_manager=pm,
            prompt_type=args.prompt_type,
        )
        responses = generate_all_responses_batch(
            pipe=pipe,
            prompts=prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        df_response = create_response_dataframe(
            dataset_df=dataset[args.split],
            responses=responses,
            config={
                "prompt_type": args.prompt_type,
                "response_type": args.response_type,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "batch_size": args.batch_size,
            },
        )
        df_response.to_csv(cache_file, index=False)
        print(f"Saved responses to {cache_file}")

    truth_col = "original_answers" if args.response_type == "id" else "natural_lang_answers"
    print(f"\nEvaluating ({truth_col})")
    evaluator = BaseEvaluator(df_response, truth_col=truth_col, pred_col="model_response")
    evaluator.evaluate_exact_partial(id=(args.response_type == "id"))


if __name__ == "__main__":
    main()
