#!/usr/bin/env python3
"""Q-LoRA fine-tuning of Llama-3.2-3B-Instruct on the temporal QA task.

Run:
    python scripts/run_finetune.py --prompt-type standard_nl --response-type nl
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quasar_rag.config import CACHE_DIR, DATA_PATH
from quasar_rag.data import load_dataset
from quasar_rag.finetuning import fine_tune_model
from quasar_rag.llm import clean_model_reload
from quasar_rag.prompting import PromptManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument(
        "--prompt-type", default="standard_nl", choices=["standard_nl", "standard_id"]
    )
    parser.add_argument("--response-type", default="nl", choices=["nl", "id"])
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save the LoRA adapter. Defaults to data/cache/qlora-<response-type>/",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        CACHE_DIR,
        f"qlora-{'natural_lang_answers' if args.response_type == 'nl' else 'original_answers'}",
    )
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset(args.data_path)
    pm = PromptManager()

    print("Reloading base quantized model...")
    model, tokenizer = clean_model_reload()

    print(
        f"Starting Q-LoRA fine-tuning "
        f"(prompt_type={args.prompt_type}, response_type={args.response_type}, "
        f"num_shots={args.num_shots})..."
    )
    fine_tune_model(
        model=model,
        tokenizer=tokenizer,
        dataset_dict=dataset,
        prompt_manager=pm,
        prompt_type=args.prompt_type,
        response_type=args.response_type,
        num_shots=args.num_shots,
        output_dir=output_dir,
    )
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
