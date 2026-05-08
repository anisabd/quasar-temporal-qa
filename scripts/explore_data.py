#!/usr/bin/env python3
"""Quick descriptive statistics over the temporal QA dataset.

Run:
    python scripts/explore_data.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

# Make package importable when running the script directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quasar_rag.config import DATA_PATH, SPLITS
from quasar_rag.data import load_dataset, load_facts_dataframe


def main():
    parser = argparse.ArgumentParser(description="Dataset exploration / EDA.")
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--save-dir", default="docs/figures")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = load_dataset(args.data_path)
    facts_dataframe = load_facts_dataframe(args.data_path)

    # 1) Random sample per split
    for split in SPLITS:
        print(f"\n{split.upper()} - random sample")
        print(dataset[split].sample(n=1).to_string())

    # 2) Counts
    for split in SPLITS:
        print(f"{split.upper()}: {len(dataset[split])} questions")
    print(f"Unique facts: {len(facts_dataframe)}")

    # 3) Unique entities
    all_entities = set()
    for split in SPLITS:
        for entities_list in dataset[split]["entities"]:
            try:
                all_entities.update(eval(entities_list) if isinstance(entities_list, str) else entities_list)
            except Exception:
                pass
    print(f"Unique entities: {len(all_entities)}")

    # 4) Question-type distribution (train + val)
    train_df = dataset["train"][["type"]].copy()
    train_df["split"] = "Train"
    val_df = dataset["val"][["type"]].copy()
    val_df["split"] = "Val"
    combined = sns.barplot  # noqa: F841 (just to ensure seaborn is loaded)

    import pandas as pd  # local import to avoid putting it at top

    combined_df = pd.concat([train_df, val_df])
    plt.figure(figsize=(10, 6))
    sns.countplot(data=combined_df, x="type", hue="split")
    plt.title("Question type distribution (train vs val)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    out = os.path.join(args.save_dir, "question_type_distribution.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

    # 5) Word counts of natural-language answers
    word_counts = []
    for answers_list in dataset["train"]["natural_lang_answers"]:
        try:
            answers = eval(answers_list) if isinstance(answers_list, str) else answers_list
        except Exception:
            answers = []
        for ans in answers:
            word_counts.append(len(str(ans).split()))
    plt.figure(figsize=(10, 6))
    plt.hist(word_counts, bins=30)
    plt.xlabel("Words per natural-language answer")
    plt.ylabel("Count")
    plt.title("Distribution of answer lengths (train)")
    plt.tight_layout()
    out = os.path.join(args.save_dir, "answer_length_distribution.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

    # 6) Train/val vs test leakage
    train_val_q = set(dataset["train"]["question"]).union(dataset["val"]["question"])
    test_q = set(dataset["test"]["question"])
    print(f"Train+Val ∩ Test = {len(train_val_q.intersection(test_q))} duplicate questions")


if __name__ == "__main__":
    main()
