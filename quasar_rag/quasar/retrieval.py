"""Enriched query encoding + parallel FAISS retrieval."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm


# ----------------------------------------------------------------------
# Query encoding
# ----------------------------------------------------------------------
def encode_question_with_entities(
    question: str,
    model,
    entity_ids: List[str],
    entity2emb_dict: Dict[str, list],
    nl_entity_ids: List[str],
    nlentity2emb_dict: Dict[str, list],
    relations: Optional[List[str]] = None,
    relation2emb_dict: Optional[Dict[str, np.ndarray]] = None,
    time_values: Optional[List[Any]] = None,
    time2emb_dict: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Build an enriched embedding: [question, mean(QIDs), mean(NL-entities), mean(times)]."""
    q_emb = model.encode(question, convert_to_numpy=True).astype("float32")

    def mean_or_zero(vecs, ref):
        if not vecs:
            return np.zeros_like(ref)
        return np.vstack(vecs).mean(axis=0)

    # QID entities
    ent_vecs = []
    for eid in entity_ids:
        if eid in entity2emb_dict:
            ent_vecs.append(np.array(entity2emb_dict[eid], dtype="float32"))
        else:
            ent_vecs.append(model.encode(eid, convert_to_numpy=True).astype("float32"))
    ent_mean = mean_or_zero(ent_vecs, q_emb)

    # NL entities
    nl_vecs = []
    for nlid in nl_entity_ids:
        if nlid in nlentity2emb_dict:
            nl_vecs.append(np.array(nlentity2emb_dict[nlid], dtype="float32"))
        else:
            nl_vecs.append(model.encode(nlid, convert_to_numpy=True).astype("float32"))
    nl_mean = mean_or_zero(nl_vecs, q_emb)

    # Temporal values
    time_values = time_values or []
    time2emb_dict = time2emb_dict or {}
    time_vecs = []
    for tval in time_values:
        if tval in time2emb_dict:
            time_vecs.append(np.array(time2emb_dict[tval], dtype="float32"))
        else:
            time_vecs.append(model.encode(str(tval), convert_to_numpy=True).astype("float32"))
    time_mean = mean_or_zero(time_vecs, q_emb)

    fused = np.concatenate([q_emb, ent_mean, nl_mean, time_mean]).astype("float32")
    fused /= np.linalg.norm(fused) + 1e-12
    return fused


# ----------------------------------------------------------------------
# Batched FAISS retrieval
# ----------------------------------------------------------------------
def batch_faiss_retrieve(
    questions: List[str],
    sentence_model,
    faiss_index,
    triples: List[dict],
    lst_qids: List[List[str]],
    entity2emb_dict: Dict[str, list],
    lst_nl_entities: List[List[str]],
    nlentity2emb_dict: Dict[str, list],
    lst_times: Optional[List[List[Any]]] = None,
    batch_size: int = 16,
    top_k: int = 3,
) -> List[List[dict]]:
    """Encode each question with its entities/times and retrieve top_k triples."""
    if lst_times is None:
        lst_times = [[] for _ in questions]

    all_retrieved: List[List[dict]] = []
    for i in tqdm(range(0, len(questions), batch_size), desc="FAISS retrieval"):
        batch_q = questions[i : i + batch_size]
        batch_qid = lst_qids[i : i + batch_size]
        batch_nl = lst_nl_entities[i : i + batch_size]
        batch_time = lst_times[i : i + batch_size]

        batch_embeddings = []
        for q, qids, nl_ids, time_s in zip(batch_q, batch_qid, batch_nl, batch_time):
            emb = encode_question_with_entities(
                question=q,
                model=sentence_model,
                entity_ids=qids,
                entity2emb_dict=entity2emb_dict,
                nl_entity_ids=nl_ids,
                nlentity2emb_dict=nlentity2emb_dict,
                time_values=time_s,
            )
            batch_embeddings.append(emb)
        batch_embeddings = np.array(batch_embeddings, dtype="float32")

        _, indices = faiss_index.search(batch_embeddings, top_k)
        for row in indices:
            all_retrieved.append([triples[j] for j in row])
    return all_retrieved


def run_parallel_faiss(
    questions,
    sentence_model,
    faiss_index,
    triples,
    lst_entities,
    entity2emb_dict,
    lst_nl_entities,
    nlentity2emb_dict,
    lst_times=None,
    batch_size: int = 16,
    top_k: int = 3,
    num_workers: int = 8,
    chunk_size: int = 256,
) -> List[List[dict]]:
    """Parallelize batch_faiss_retrieve over chunks of questions."""
    if lst_times is None:
        lst_times = [[] for _ in questions]

    chunks = []
    for i in range(0, len(questions), chunk_size):
        chunks.append(
            (
                questions[i : i + chunk_size],
                lst_entities[i : i + chunk_size],
                lst_nl_entities[i : i + chunk_size],
                lst_times[i : i + chunk_size],
            )
        )

    def _process(chunk):
        q, e, nl, t = chunk
        return batch_faiss_retrieve(
            q,
            sentence_model,
            faiss_index,
            triples,
            e,
            entity2emb_dict,
            nl,
            nlentity2emb_dict,
            lst_times=t,
            batch_size=batch_size,
            top_k=top_k,
        )

    results: List[List[dict]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for out in executor.map(_process, chunks):
            results.extend(out)
    return results


# ----------------------------------------------------------------------
# Prompt augmentation (thread-safe variant)
# ----------------------------------------------------------------------
def add_facts_to_prompt_ts(
    messages: List[Dict[str, str]], retrieved_facts: List[Any]
) -> List[Dict[str, str]]:
    """Deep-copy version of add_facts_to_prompt suitable for ThreadPoolExecutor."""
    messages = copy.deepcopy(messages)
    user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            user_idx = i
            break
    if user_idx is None:
        return messages

    facts_block = "Here is some information to help you answering the question:\n"
    facts_block += "\n".join(f"- {fact}" for fact in retrieved_facts)
    messages[user_idx]["content"] += "\n\n" + facts_block
    return messages


def merge_prompt_with_facts(prompt_dict: Dict[str, Any], facts: List[Any]) -> Dict[str, Any]:
    prompt_dict["prompt"] = add_facts_to_prompt_ts(prompt_dict["prompt"], facts)
    return prompt_dict
