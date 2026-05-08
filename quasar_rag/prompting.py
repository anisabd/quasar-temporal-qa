"""PromptManager and prompt-construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class PromptManager:
    """Holds system prompts and helpers to build chat-formatted messages."""

    basic: str = """You are a helpful question answering assistant.
Answer the question based on the provided information.
Provide only the answer in a clear and concise format."""

    standard_nl: str = """You are a helpful question answering assistant.
Your task is to answer user questions in natural language, clearly and concisely.
- Always answer in complete sentences.
- If the answer is uncertain or ambiguous, acknowledge it explicitly using for example
expressions like "It is unclear" or "There is not enough information")."""

    standard_id: str = """You are a classification assistant.
Your task is to provide the correct identifier (ID) corresponding to the user's question.
- Answer ONLY with the correct ID value or choose the one that best fits the question.
- No need to explain or add additional informations.
- If the answer cannot be determined, respond with "UNKNOWN"."""

    new_standard_nl: str = """You are a precise and concise assistant.
Answer each question with the **exact short answer** only — no explanation.

- If the answer is a name, give the full name only.
- If it's a number, date, or place, output that only.
- If the question asks about a date, **always provide only the year** unless explicitly asked for a specific day or month.
- Use a comma-separated list if there are multiple valid answers.
- Do not use sentences, punctuation, or filler words."""

    def __post_init__(self):
        self.prompts: Dict[str, str] = {
            "basic": self.basic,
            "standard_nl": self.standard_nl,
            "standard_id": self.standard_id,
            "standard_nl2": self.new_standard_nl,
        }

    # ------------------------------------------------------------------
    # Few-shot
    # ------------------------------------------------------------------
    def _get_few_shot_examples(
        self,
        train_df: Optional[pd.DataFrame],
        num_shots: int,
        response_type: str,
    ) -> List[Tuple[str, str]]:
        """Return up to `num_shots` (question, answer) pairs picked from a curated id list."""
        few_shot_ids = [3043, 5696, 6149, 988, 1045, 6889, 7074, 7363, 7544, 9642]

        if train_df is None or num_shots <= 0:
            return []

        subset_df = train_df[train_df["id"].isin(few_shot_ids)]
        sampled_df = subset_df.sample(
            n=min(num_shots, len(subset_df)), random_state=42
        )

        if response_type == "nl":
            answer_col = "natural_lang_answers"
        elif response_type == "id":
            answer_col = "original_answers"
        else:
            raise ValueError("Unknown response_type")

        return list(zip(sampled_df["question"], sampled_df[answer_col]))

    # ------------------------------------------------------------------
    # Prompt creation
    # ------------------------------------------------------------------
    def create_prompt(
        self,
        query: str,
        prompt_type: str,
        response_type: str = "nl",
        num_shots: int = 0,
        train_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, str]]:
        """Build a chat-format prompt with optional few-shot examples."""
        system_prompt = self.prompts.get(prompt_type, self.basic)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        if num_shots > 0 and train_df is not None:
            examples = self._get_few_shot_examples(train_df, num_shots, response_type)
            messages.append({"role": "system", "content": "Here are a few examples:"})
            for question, answer in examples:
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})

        messages.append({"role": "user", "content": query})
        return messages

    def display_prompt_template(
        self, messages: List[Dict[str, str]], tokenizer
    ) -> None:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        print("Formatted prompt:\n")
        print(formatted_prompt)


# ----------------------------------------------------------------------
# Bulk prompt creation
# ----------------------------------------------------------------------
def create_all_prompts(
    dataset_df: pd.DataFrame,
    prompt_manager: PromptManager,
    prompt_type: str = "standard_id",
    num_shots: int = 0,
    train_df: Optional[pd.DataFrame] = None,
) -> List[List[Dict[str, str]]]:
    """Create prompts for every question in the dataframe."""
    print(f"Creating prompts for {len(dataset_df)} questions...")
    questions = dataset_df["question"].to_list()
    response_type = "id" if prompt_type == "standard_id" else "nl"

    all_prompts = []
    for q in questions:
        prompt = prompt_manager.create_prompt(
            query=q,
            prompt_type=prompt_type,
            response_type=response_type,
            num_shots=num_shots,
            train_df=train_df,
        )
        all_prompts.append(prompt)
    print(f"Created {len(all_prompts)} prompts")
    return all_prompts


def add_facts_to_prompt(
    messages: List[Dict[str, str]], retrieved_facts: List[str]
) -> List[Dict[str, str]]:
    """Append retrieved-facts block to the last user message."""
    messages = messages.copy()
    facts_text = "\nHere is some information to help you answering the question:\n"
    facts_text += "\n".join([f"{i}. {fact}" for i, fact in enumerate(retrieved_facts, 1)])
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages[i]["content"] += facts_text
            break
    return messages


def create_response_dataframe(
    dataset_df: pd.DataFrame,
    responses: List[Any],
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    df_response = dataset_df.copy()
    df_response["model_response"] = responses
    if config:
        for key, value in config.items():
            df_response[f"config_{key}"] = value
    return df_response
