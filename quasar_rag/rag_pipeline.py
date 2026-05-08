"""End-to-end baseline RAG evaluation pipeline (Section 4 of the report)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .config import DATA_PATH
from .evaluation import BaseEvaluator
from .llm import generate_all_responses_batch
from .prompting import (
    PromptManager,
    add_facts_to_prompt,
    create_all_prompts,
    create_response_dataframe,
)
from .retriever import CustomRetriever, batch_retrieve


def run_rag_evaluation(
    dataset: Dict[str, pd.DataFrame],
    retriever: CustomRetriever,
    facts_dataframe: pd.DataFrame,
    pm: PromptManager,
    pipe,
    top_k_values: Optional[List[int]] = None,
    prompting_config: Optional[Dict[str, Any]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    data_path: str = DATA_PATH,
    split: str = "val",
) -> Dict[int, Dict[str, Any]]:
    """Run the RAG pipeline (retrieve → augment → generate → evaluate) for several top_k values."""
    if top_k_values is None:
        top_k_values = [1, 3, 5]
    if prompting_config is None:
        prompting_config = {
            "prompt_type": "standard_id",
            "num_shots": 10,
        }
    if generation_config is None:
        generation_config = {
            "max_new_tokens": 128,
            "temperature": 0.3,
            "top_p": 0.8,
            "batch_size": 16,
        }

    is_id = prompting_config["prompt_type"] == "standard_id"
    truth_col = "original_answers" if is_id else "natural_lang_answers"

    results: Dict[int, Dict[str, Any]] = {}

    for top_k in top_k_values:
        print(f"\n{'=' * 80}")
        print(f"EVALUATION WITH RETRIEVED FACTS (TOP_K = {top_k})")
        print(f"{'=' * 80}\n")

        # 1. Retrieval (cached)
        retrieval_cache = os.path.join(
            data_path, "cache", f"{split}_data_retrieved_fact_cache_top{top_k}.csv"
        )
        if os.path.exists(retrieval_cache):
            print(f"Loading cached retrieved facts (top-{top_k})...")
            df_split = pd.read_csv(retrieval_cache)
        else:
            print(f"Running batch_retrieve with top_k={top_k}...")
            df_split = dataset[split].copy()
            results_ids, results_texts = batch_retrieve(
                df_split["question"].to_list(),
                retriever,
                facts_dataframe,
                batch_size=16,
                top_k=top_k,
            )
            df_split["retrieved_facts_ID"] = results_ids
            df_split["retrieved_facts_Text"] = results_texts
            df_split.to_csv(retrieval_cache, index=False)
            print(f"Saved to {retrieval_cache}")

        # 2. Generation (cached)
        facts_type = "retrieved"
        prompt_tag = prompting_config["prompt_type"].split("_")[1]
        response_cache = os.path.join(
            data_path,
            "cache",
            f"{split}_data_model_response_{facts_type}_top{top_k}_{prompt_tag}.csv",
        )

        if os.path.exists(response_cache):
            print(f"Loading cached responses (top-{top_k})...")
            df_response = pd.read_csv(response_cache)
        else:
            print(f"Generating prompts for {split} dataset...")
            split_prompts = create_all_prompts(
                dataset_df=dataset[split],
                prompt_manager=pm,
                **prompting_config,
            )

            print(f"Adding top-{top_k} facts to prompts...")
            for i, row in df_split.iterrows():
                split_prompts[i] = add_facts_to_prompt(
                    split_prompts[i], row["retrieved_facts_Text"]
                )

            print("Generating responses...")
            split_responses = generate_all_responses_batch(
                pipe=pipe, prompts=split_prompts, **generation_config
            )

            full_config = {
                **prompting_config,
                **generation_config,
                "top_k": top_k,
                "facts_type": facts_type,
            }
            df_response = create_response_dataframe(
                dataset_df=dataset[split],
                responses=split_responses,
                config=full_config,
            )
            df_response.to_csv(response_cache, index=False)
            print(f"Saved to {response_cache}")

        # 3. Evaluation
        print(f"\nEvaluating with top_k={top_k}...")
        evaluator = BaseEvaluator(
            df_response, truth_col=truth_col, pred_col="model_response"
        )
        df_evaluated = evaluator.evaluate_exact_partial(id=is_id)
        results[top_k] = {
            "df_evaluated": df_evaluated,
            "exact_precision": df_evaluated["precision_exact"].mean(),
            "exact_recall": df_evaluated["recall_exact"].mean(),
            "exact_f1": df_evaluated["f1_exact"].mean(),
            "partial_precision": df_evaluated["precision_partial"].mean(),
            "partial_recall": df_evaluated["recall_partial"].mean(),
            "partial_f1": df_evaluated["f1_partial"].mean(),
        }
    return results


def display_comparison_results(results: Dict[int, Dict[str, Any]]) -> None:
    """Pretty-print metrics across top_k values."""
    print(f"\n{'=' * 100}")
    print("COMPARISON OF RESULTS ACROSS DIFFERENT TOP_K VALUES")
    print(f"{'=' * 100}\n")

    header = f"{'Top_K':<10}"
    for metric in ["Precision", "Recall", "F1"]:
        header += f"{'Exact ' + metric:<20}{'Partial ' + metric:<20}"
    print(header)
    print("-" * 100)

    for top_k in sorted(results.keys()):
        row = f"{top_k:<10}"
        row += f"{results[top_k]['exact_precision']:.4f}{'':<15}"
        row += f"{results[top_k]['partial_precision']:.4f}{'':<15}"
        row += f"{results[top_k]['exact_recall']:.4f}{'':<15}"
        row += f"{results[top_k]['partial_recall']:.4f}{'':<15}"
        row += f"{results[top_k]['exact_f1']:.4f}{'':<15}"
        row += f"{results[top_k]['partial_f1']:.4f}{'':<15}"
        print(row)
    print("\n" + "=" * 100)

    best_exact = max(results.keys(), key=lambda k: results[k]["exact_f1"])
    best_partial = max(results.keys(), key=lambda k: results[k]["partial_f1"])
    print(
        f"\n Best top_k for Exact Match F1:   {best_exact} "
        f"(F1={results[best_exact]['exact_f1']:.4f})"
    )
    print(
        f" Best top_k for Partial Match F1: {best_partial} "
        f"(F1={results[best_partial]['partial_f1']:.4f})"
    )


def plot_comparison_results(
    results: Dict[int, Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Plot precision/recall/F1 (exact + partial) across top_k values."""
    top_k_values = sorted(results.keys())

    exact_p = [results[k]["exact_precision"] for k in top_k_values]
    exact_r = [results[k]["exact_recall"] for k in top_k_values]
    exact_f = [results[k]["exact_f1"] for k in top_k_values]
    partial_p = [results[k]["partial_precision"] for k in top_k_values]
    partial_r = [results[k]["partial_recall"] for k in top_k_values]
    partial_f = [results[k]["partial_f1"] for k in top_k_values]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(top_k_values, exact_p, marker="o", label="Precision", linewidth=2)
    axes[0].plot(top_k_values, exact_r, marker="s", label="Recall", linewidth=2)
    axes[0].plot(top_k_values, exact_f, marker="^", label="F1", linewidth=2)
    axes[0].set_xlabel("Top K", fontsize=12)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].set_title("Exact Match Metrics vs Top K", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(top_k_values)

    axes[1].plot(top_k_values, partial_p, marker="o", label="Precision", linewidth=2)
    axes[1].plot(top_k_values, partial_r, marker="s", label="Recall", linewidth=2)
    axes[1].plot(top_k_values, partial_f, marker="^", label="F1", linewidth=2)
    axes[1].set_xlabel("Top K", fontsize=12)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title("Partial Match Metrics vs Top K", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(top_k_values)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
