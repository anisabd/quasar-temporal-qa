"""Loading + QID-enrichment of the temporal-triple corpus."""

from __future__ import annotations

import json
import os
from typing import List

import pandas as pd

from ..config import DATA_PATH


def load_triples_dataframe(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Read facts.json into a clean DataFrame of triples."""
    with open(os.path.join(data_path, "facts.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    new_list = [d for d in data if isinstance(d, dict)]
    df = pd.DataFrame(new_list)
    df = df[df["subject"].notna()]
    df = df[df["object"].notna()]
    return df


def build_facts_plus(
    df: pd.DataFrame,
    data_path: str = DATA_PATH,
) -> List[dict]:
    """Augment each triple with subject_qid / object_qid using label_mappings.json."""
    facts_with_qid_file = os.path.join(data_path, "facts_plus.json")
    if os.path.exists(facts_with_qid_file):
        with open(facts_with_qid_file, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(os.path.join(data_path, "label_mappings.json"), "r", encoding="utf-8") as f:
        lbl_data = json.load(f)

    name_to_qid = {v: k for k, v in lbl_data.items()}
    df = df.copy()
    df["subject_qid"] = df["subject"].map(name_to_qid)
    df["object_qid"] = df["object"].map(name_to_qid)

    records = df.to_dict(orient="records")
    clean_records = [
        {k: v for k, v in rec.items() if pd.notna(v)} for rec in records
    ]

    with open(facts_with_qid_file, "w", encoding="utf-8") as f:
        json.dump(clean_records, f, ensure_ascii=False, indent=2)

    return clean_records
