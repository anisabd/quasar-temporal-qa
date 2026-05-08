"""Multi-threaded prompt creation + LLM generation used by QUASAR."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from ..prompting import PromptManager


def create_all_prompts_parallel(
    dataset_df: pd.DataFrame,
    prompt_manager: PromptManager,
    prompt_type: str = "standard_id",
    num_shots: int = 0,
    train_df: Optional[pd.DataFrame] = None,
    max_workers: int = 16,
) -> List[Dict[str, Any]]:
    """Parallel version of create_all_prompts; preserves per-row id."""
    questions = dataset_df["question"].to_list()
    ids = dataset_df["id"].to_list()
    response_type = "id" if prompt_type == "standard_id" else "nl"

    def _create_single_prompt(pair):
        qid, qtext = pair
        prompt = prompt_manager.create_prompt(
            query=qtext,
            prompt_type=prompt_type,
            response_type=response_type,
            num_shots=num_shots,
            train_df=train_df,
        )
        return {"id": qid, "prompt": prompt}

    print(f"Creating prompts with threading ({max_workers} workers)...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_create_single_prompt, zip(ids, questions)))
    print(f"Created {len(results)} prompts")
    return results


def generate_responses_parallel(
    pipe,
    prompts_with_ids: List[Dict[str, Any]],
    batch_size: int = 16,
    max_workers: int = 8,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.8,
    **generation_kwargs,
) -> List[Dict[str, Any]]:
    """Threaded batched generation; returns [{'id': ..., 'response': [...]}, ...]."""

    def _generate_batch(batch):
        batch_prompts = [item["prompt"] for item in batch]
        batch_ids = [item["id"] for item in batch]
        batch_outputs = pipe(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **generation_kwargs,
        )
        batch_responses = []
        for qid, output in zip(batch_ids, batch_outputs):
            messages = output[0].get("generated_text", "")
            last_msg = next(
                (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
                "",
            )
            batch_responses.append({"id": qid, "response": [last_msg]})
        return batch_responses

    batches = [
        prompts_with_ids[i : i + batch_size]
        for i in range(0, len(prompts_with_ids), batch_size)
    ]

    all_results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_batch, batch): idx
            for idx, batch in enumerate(batches)
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Generating batches"):
            all_results.extend(f.result())
    return all_results


def create_merged_dataframe(
    dataset_df: pd.DataFrame,
    responses_with_ids: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Merge per-id responses back onto the original dataframe."""
    df_resp = pd.DataFrame(responses_with_ids).rename(
        columns={"response": "model_response"}
    )
    merged_df = dataset_df.merge(df_resp, on="id", how="left")
    if config:
        for key, value in config.items():
            merged_df[f"config_{key}"] = value
    return merged_df
