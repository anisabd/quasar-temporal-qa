"""Evaluation utilities (Exact Match / Partial Match F1)."""

from __future__ import annotations

import ast
import string
from typing import List, Union

import numpy as np
import pandas as pd


def normalize_text(text: str) -> str:
    """Lowercase + strip simple punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def compare_entities(str1: str, str2: str, method: str = "exact") -> bool:
    """Compare two entity strings under either 'exact' or 'partial' matching."""
    str1_norm = normalize_text(str1)
    str2_norm = normalize_text(str2)
    if method == "exact":
        return str1_norm == str2_norm
    if method == "partial":
        return str1_norm in str2_norm or str2_norm in str1_norm
    raise ValueError("Unknown method, use 'exact' or 'partial'")


class BaseEvaluator:
    """Compute precision / recall / F1 between gold and predicted entity lists."""

    def __init__(
        self,
        df: pd.DataFrame,
        truth_col: str = "natural_lang_answers",
        pred_col: str = "model_response",
    ):
        self.df = df.copy()
        self.truth_col = truth_col
        self.pred_col = pred_col

    @staticmethod
    def _normalize_text(text: str) -> str:
        return normalize_text(text)

    @staticmethod
    def _parse_list_string(list_str: str) -> List[str]:
        stripped = list_str.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            return []
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        try:
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
            return [str(parsed)]
        except (SyntaxError, ValueError):
            items, current_item = [], []
            in_quotes, quote_char = False, None
            for char in inner:
                if char in ('"', "'") and (not in_quotes or char == quote_char):
                    if in_quotes:
                        in_quotes, quote_char = False, None
                    else:
                        in_quotes, quote_char = True, char
                elif char == "," and not in_quotes:
                    if current_item:
                        item = "".join(current_item).strip().strip('"').strip("'")
                        if item:
                            items.append(item)
                    current_item = []
                else:
                    current_item.append(char)
            if current_item:
                item = "".join(current_item).strip().strip('"').strip("'")
                if item:
                    items.append(item)
            return items

    def _to_list(self, value: Union[str, List[str]]) -> List[str]:
        if isinstance(value, str):
            try:
                parsed = self._parse_list_string(value)
                return [self._normalize_text(item) for item in parsed]
            except Exception as e:
                print(f"Failed to parse '{value}': {e}")
                return []
        if isinstance(value, list):
            return [self._normalize_text(str(v)) for v in value]
        print(f"Unexpected type for value: {type(value)}")
        return []

    def _compare_entities(self, ref: str, other: str, method: str = "exact") -> bool:
        return compare_entities(ref, other, method)

    def evaluate(self, criteria: str):
        """Evaluate predictions under the given criteria ('exact' or 'partial')."""
        precisions, recalls, f1s = [], [], []
        for _, row in self.df.iterrows():
            truths = self._to_list(row[self.truth_col])
            preds = self._to_list(row[self.pred_col])

            true_positives = 0
            matched_truths = set()
            for pred in preds:
                for i, truth in enumerate(truths):
                    if i not in matched_truths and self._compare_entities(
                        truth, pred, method=criteria
                    ):
                        true_positives += 1
                        matched_truths.add(i)
                        break

            precision = true_positives / len(preds) if preds else 0
            recall = true_positives / len(truths) if truths else 0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        self.df[f"precision_{criteria}"] = precisions
        self.df[f"recall_{criteria}"] = recalls
        self.df[f"f1_{criteria}"] = f1s

        return {
            "avg_precision": float(np.mean(precisions)),
            "avg_recall": float(np.mean(recalls)),
            "avg_f1": float(np.mean(f1s)),
            "df": self.df,
        }

    def evaluate_exact_partial(self, id: bool = False) -> pd.DataFrame:
        """Run both 'exact' and 'partial' evaluations and pretty-print results."""
        results = {}
        for criteria in ["exact", "partial"]:
            res = self.evaluate(criteria=criteria)
            results[criteria] = res
            self.df = res["df"]

        print()
        if not id:
            print(f"{'Metric':<15} {'Exact':<15} {'Partial':<15}")
            print("-" * 45)
            print(
                f"{'Precision':<15} "
                f"{results['exact']['avg_precision']:.2f}{'':<11} "
                f"{results['partial']['avg_precision']:.2f}"
            )
            print(
                f"{'Recall':<15} "
                f"{results['exact']['avg_recall']:.2f}{'':<11} "
                f"{results['partial']['avg_recall']:.2f}"
            )
            print(
                f"{'F1':<15} "
                f"{results['exact']['avg_f1']:.2f}{'':<11} "
                f"{results['partial']['avg_f1']:.2f}"
            )
        else:
            print(f"{'Metric':<15} {'Exact':<15}")
            print("-" * 45)
            print(
                f"{'Precision':<15} "
                f"{results['exact']['avg_precision']:.2f}"
            )
            print(
                f"{'Recall':<15} "
                f"{results['exact']['avg_recall']:.2f}"
            )
            print(f"{'F1':<15} {results['exact']['avg_f1']:.2f}")

        return self.df
