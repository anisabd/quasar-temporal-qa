"""End-to-end QUASAR evaluation: enriched-FAISS retrieval + parallel generation."""

from __future__ import annotations

import ast
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import pandas as pd

from ..config import DATA_PATH
from ..evaluation import BaseEvaluator
from ..prompting import PromptManager
from .parallel_generation import (
    create_all_prompts_parallel,
    create_merged_dataframe,
    generate_responses_parallel,
)
from .retrieval import merge_prompt_with_facts, run_parallel_faiss


def _safe_literal_eval(x):
    return ast.literal_eval(x) if isinstance(x, str) else []


def run_quasar_evaluation(
    dataset: Dict[str, pd.DataFrame],
    sentence_model,
    faiss_index,
    triples,
    entity2emb: Dict[str, list],
    nlentity2emb: Dict[str, list],
    pm: PromptManager,
    pipe,
    top_k: int = 5,
    prompting_config: Optional[Dict[str, Any]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    data_path: str = DATA_PATH,
    split: str = "val",
    cache_suffix: str = "_V2",
) -> Dict[str, Dict[str, Any]]:
    """Run QUASAR retrieval → augment → generate → evaluate over a split."""
    if prompting_config is None:
        prompting_config = {"prompt_type": "standard_id"}
    if generation_config is None:
        generation_config = {
            "max_new_tokens": 128,
            "temperature": 0.3,
            "top_p": 0.8,
            "batch_size": 16,
        }

    truth_col = "original_answers"

    print(f"\n{'=' * 80}")
    print(f"QUASAR EVALUATION (TOP_K = {top_k})")
    print(f"{'=' * 80}\n")

    # 1. Retrieval (cached)
    retrieval_cache = os.path.join(
        data_path, "cache", f"{split}_data_retrieved_fact_cache_top{top_k}{cache_suffix}.csv"
    )
    if os.path.exists(retrieval_cache):
        print(f"Loading cached retrieved facts (top-{top_k})...")
        df_split = pd.read_csv(retrieval_cache)
    else:
        print(f"Running QUASAR FAISS retrieval with top_k={top_k}...")
        df_split = dataset[split].copy()
        df_split["entities"] = df_split["entities"].apply(_safe_literal_eval)
        df_split["natural_lang_entities"] = df_split["natural_lang_entities"].apply(
            _safe_literal_eval
        )
        df_split["times"] = df_split["times"].apply(_safe_literal_eval)

        results_texts = run_parallel_faiss(
            df_split["question"].tolist(),
            sentence_model,
            faiss_index,
            triples,
            df_split["entities"].tolist(),
            entity2emb,
            df_split["natural_lang_entities"].tolist(),
            nlentity2emb,
            lst_times=df_split["times"].tolist(),
            batch_size=16,
            top_k=top_k,
        )
        df_split["retrieved_facts_Text"] = results_texts
        df_split.to_csv(retrieval_cache, index=False)
        print(f"Saved retrieval cache to {retrieval_cache}")

    # 2. Generation (cached)
    facts_type = "retrieved"
    prompt_tag = prompting_config["prompt_type"].split("_")[1]
    response_cache = os.path.join(
        data_path,
        "cache",
        f"{split}_data_model_response_{facts_type}_top{top_k}_{prompt_tag}{cache_suffix}.csv",
    )

    if os.path.exists(response_cache):
        print(f"Loading cached responses (top-{top_k})...")
        df_response = pd.read_csv(response_cache)
    else:
        print(f"Generating prompts (parallel)...")
        split_prompts = create_all_prompts_parallel(
            dataset_df=dataset[split], prompt_manager=pm
        )
        print(f"Adding top-{top_k} facts to prompts...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            split_prompts = list(
                ex.map(
                    merge_prompt_with_facts,
                    split_prompts,
                    df_split["retrieved_facts_Text"],
                )
            )
        print("Generating responses (parallel)...")
        split_responses = generate_responses_parallel(
            pipe=pipe,
            prompts_with_ids=split_prompts,
            batch_size=16,
            max_workers=16,
            max_new_tokens=128,
        )

        full_config = {
            **prompting_config,
            **generation_config,
            "top_k": top_k,
            "facts_type": facts_type,
        }
        df_response = create_merged_dataframe(
            dataset_df=dataset[split],
            responses_with_ids=split_responses,
            config=full_config,
        )
        df_response.to_csv(response_cache, index=False)
        print(f"Saved responses to {response_cache}")

    # 3. Evaluation
    print(f"\nEvaluating with top_k={top_k}...")
    evaluator = BaseEvaluator(
        df_response, truth_col=truth_col, pred_col="model_response"
    )
    df_evaluated = evaluator.evaluate_exact_partial(id=True)

    return {
        f"{top_k}": {
            "df_evaluated": df_evaluated,
            "exact_precision": df_evaluated["precision_exact"].mean(),
            "exact_recall": df_evaluated["recall_exact"].mean(),
            "exact_f1": df_evaluated["f1_exact"].mean(),
            "partial_precision": df_evaluated["precision_partial"].mean(),
            "partial_recall": df_evaluated["recall_partial"].mean(),
            "partial_f1": df_evaluated["f1_partial"].mean(),
        }
    }
