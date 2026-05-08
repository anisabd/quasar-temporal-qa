"""Build/load embeddings for relations, NL entities, QIDs, and triples."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from ..config import DATA_PATH


# ----------------------------------------------------------------------
# Per-field embeddings
# ----------------------------------------------------------------------
def build_relation_embeddings(
    triples: List[dict],
    sentence_model,
    data_path: str = DATA_PATH,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """Encode every unique relation; persist to disk; return (matrix, list, dict)."""
    rel_emb_file = os.path.join(data_path, "relation_embeddings.npy")
    rel_json_file = os.path.join(data_path, "relations.json")

    if os.path.exists(rel_emb_file):
        rel_embeddings = np.load(rel_emb_file)
    else:
        unique_rel = list({t["relation"] for t in triples if t.get("relation")})
        rel_embeddings = sentence_model.encode(
            unique_rel, batch_size=32, normalize_embeddings=True
        )
        np.save(rel_emb_file, rel_embeddings)

    if os.path.exists(rel_json_file):
        with open(rel_json_file, "r", encoding="utf-8") as f:
            relations = json.load(f)
    else:
        relations = list({t["relation"] for t in triples if t.get("relation")})
        with open(rel_json_file, "w", encoding="utf-8") as f:
            json.dump(relations, f, indent=2, ensure_ascii=False)

    rel2emb = {r: rel_embeddings[i] for i, r in enumerate(relations)}
    return rel_embeddings, relations, rel2emb


def build_nl_entity_embeddings(
    triples: List[dict],
    sentence_model,
    data_path: str = DATA_PATH,
    batch_size: int = 256,
) -> Dict[str, list]:
    """Map every natural-language entity to its embedding (persisted as JSON)."""
    nl_entity_emb_file = os.path.join(data_path, "nlentity2emb.json")
    if os.path.exists(nl_entity_emb_file):
        with open(nl_entity_emb_file, "r", encoding="utf-8") as f:
            nlentity2emb = json.load(f)
        print(f"Loaded {len(nlentity2emb)} NL-entity embeddings")
        return nlentity2emb

    entities = set()
    for t in triples:
        if t.get("subject") is not None:
            entities.add(t["subject"])
        if t.get("object") is not None:
            entities.add(t["object"])
    entities = list(entities)
    print("Unique NL entities:", len(entities))

    nlentity2emb: Dict[str, list] = {}
    for i in tqdm(range(0, len(entities), batch_size), desc="Encoding NL entities"):
        batch = entities[i : i + batch_size]
        batch_embs = sentence_model.encode(batch, convert_to_numpy=True).astype("float32")
        for ent, emb in zip(batch, batch_embs):
            nlentity2emb[ent] = emb.tolist()

    with open(nl_entity_emb_file, "w", encoding="utf-8") as f:
        json.dump(nlentity2emb, f, indent=2, ensure_ascii=False)
    return nlentity2emb


def build_qid_embeddings(
    triples: List[dict],
    sentence_model,
    data_path: str = DATA_PATH,
    batch_size: int = 256,
) -> Dict[str, list]:
    """Map every Wikidata QID (as text) to its embedding."""
    entity_emb_file = os.path.join(data_path, "entity2emb.json")
    if os.path.exists(entity_emb_file):
        with open(entity_emb_file, "r", encoding="utf-8") as f:
            entity2emb = json.load(f)
        print(f"Loaded {len(entity2emb)} QID embeddings")
        return entity2emb

    entities = set()
    for t in triples:
        if t.get("subject_qid") is not None:
            entities.add(t["subject_qid"])
        if t.get("object_qid") is not None:
            entities.add(t["object_qid"])
    entities = list(entities)
    print("Unique QIDs:", len(entities))

    entity2emb: Dict[str, list] = {}
    for i in tqdm(range(0, len(entities), batch_size), desc="Encoding QIDs"):
        batch = entities[i : i + batch_size]
        batch_embs = sentence_model.encode(batch, convert_to_numpy=True).astype("float32")
        for eid, emb in zip(batch, batch_embs):
            entity2emb[eid] = emb.tolist()

    with open(entity_emb_file, "w", encoding="utf-8") as f:
        json.dump(entity2emb, f, indent=2, ensure_ascii=False)
    return entity2emb


# ----------------------------------------------------------------------
# Triple embedding
# ----------------------------------------------------------------------
def encode_entity(sentence_model, entity_label: str, entity_qid=None) -> np.ndarray:
    """Encode an entity as 'entity: <label> | id: <qid|noQID>'."""
    text = f"entity: {entity_label} | id: {entity_qid if entity_qid else 'noQID'}"
    return sentence_model.encode(text, convert_to_numpy=True).astype("float32")


def encode_date(sentence_model, date_str) -> np.ndarray:
    """Encode a date string, or return a zero vector when missing."""
    if date_str is None:
        return np.zeros(384, dtype="float32")
    return sentence_model.encode(str(date_str), convert_to_numpy=True).astype("float32")


def encode_relation(sentence_model, rel_name: str, rel2emb_dict: Dict[str, np.ndarray]):
    if rel_name in rel2emb_dict:
        return rel2emb_dict[rel_name]
    return sentence_model.encode(rel_name, convert_to_numpy=True).astype("float32")


def encode_triple(triple: dict, sentence_model, rel2emb=None) -> np.ndarray:
    """Concatenate [subject_emb, object_emb, start_date_emb, end_date_emb]."""
    h = encode_entity(sentence_model, triple["subject"], triple.get("subject_qid"))
    t = encode_entity(sentence_model, triple["object"], triple.get("object_qid"))
    st = encode_date(sentence_model, triple.get("start_time"))
    et = encode_date(sentence_model, triple.get("end_time"))
    return np.concatenate([h, t, st, et])


def build_triple_embeddings(
    triples: List[dict],
    sentence_model,
    rel2emb=None,
    data_path: str = DATA_PATH,
) -> np.ndarray:
    """Build (or load) the matrix of triple embeddings."""
    triple_emb_file = os.path.join(data_path, "triple_embeddings.npy")
    if os.path.exists(triple_emb_file):
        return np.load(triple_emb_file)

    embeddings = []
    for triple in tqdm(triples, desc="Encoding triples"):
        embeddings.append(encode_triple(triple, sentence_model, rel2emb))
    embeddings = np.array(embeddings)
    print("Triple-embeddings shape:", embeddings.shape)
    np.save(triple_emb_file, embeddings)
    return embeddings
