"""Dataset loading utilities for the temporal QA task."""

from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd

from .config import DATA_PATH, SPLITS


def load_dataset(data_path: str = DATA_PATH) -> Dict[str, pd.DataFrame]:
    """Load train/val/test CSVs into a dict of DataFrames."""
    return {
        split: pd.read_csv(os.path.join(data_path, f"{split}.csv"))
        for split in SPLITS
    }


def load_facts_dataframe(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load the flat facts file (one fact per row, with an integer id)."""
    facts_path = os.path.join(data_path, "facts.json")
    with open(facts_path, "r", encoding="utf-8") as f:
        list_of_facts = [json.loads(line) for line in f][0]
    return pd.DataFrame(
        {
            "id": list(range(len(list_of_facts))),
            "fact": [str(fact) for fact in list_of_facts],
        }
    )


def load_label_mapping_dataframe(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load the QID -> label mapping as a DataFrame."""
    with open(os.path.join(data_path, "label_mappings.json")) as f:
        data = json.load(f)
    return pd.DataFrame(list(data.items()), columns=["id", "label"])


# ----------------------------------------------------------------------
# Helpers from the original notebook (Section 0.1)
# ----------------------------------------------------------------------
def get_factID_from_factLabel(label, facts_dataframe: pd.DataFrame):
    label_str = str(label)
    result = facts_dataframe[facts_dataframe["fact"] == label_str]["id"].to_list()
    if not result:
        return None
    return result[0]


def get_factLabel_from_factID(fact_id, facts_dataframe: pd.DataFrame):
    return facts_dataframe[facts_dataframe["id"] == fact_id]["fact"].to_list()[0]
