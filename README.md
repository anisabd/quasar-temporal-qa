# QUASAR — Temporal Question Answering with Retrieval-Augmented Generation

QUASAR (**Q**uestion-Answering with **U**nified **S**tructured **A**nd **R**etrieval-augmented embeddings) is a Retrieval-Augmented Generation (RAG) pipeline for **temporal question answering**. It pairs a 4-bit-quantized **Llama-3.2-3B-Instruct** generator with a **FAISS** index of QID-enriched temporal triples, and retrieves with an enriched query embedding that fuses the question, its associated entities (both Wikidata IDs and natural-language labels), and the temporal values mentioned in the question.

The repository is a clean adaptation of an academic NLP project (INF8460 — Polytechnique Montréal) into a runnable, reproducible codebase. It implements four progressively stronger systems on the same dataset and evaluation harness.

| Stage | What it does | Entry point |
|---|---|---|
| Parametric memory | LLM-only baseline, no retrieval | `scripts/run_parametric.py` |
| Vanilla RAG | MiniLM embeddings of flat facts + cosine retrieval | `scripts/run_rag.py` |
| **QUASAR** | FAISS index of structured triples + enriched query embeddings | `scripts/run_quasar.py` |
| Q-LoRA | 4-bit QLoRA fine-tuning of the LLM | `scripts/run_finetune.py` |

## Method overview

QUASAR does **not** reconstruct a full knowledge graph. Instead it indexes each temporal triple `(subject, relation, object, start_time, end_time)` — together with its Wikidata QIDs — as a dense vector in a FAISS `IndexFlatIP` (cosine similarity over L2-normalized vectors). At query time, the question embedding is concatenated with mean-pooled embeddings of the question's entities, NL entities and temporal values, producing a single retrieval vector that captures both lexical and structural cues. The top-k retrieved triples are then injected into the LLM prompt as context, and the LLM generates the final answer (entity ID or natural language).

The triples corpus is built from `facts.json` and enriched with QIDs via `label_mappings.json`. Per-field embeddings (relations, NL entities, QIDs, dates) are precomputed and cached so subsequent runs are fast.

## Repository layout

```
quasar-temporal-qa/
├── quasar_rag/
│   ├── config.py              # Paths, model IDs, splits
│   ├── data.py                # CSV / facts / mapping loaders
│   ├── evaluation.py          # Exact / partial F1 evaluator
│   ├── prompting.py           # PromptManager + chat-format helpers
│   ├── llm.py                 # 4-bit Llama loader + generation
│   ├── retriever.py           # MiniLM cosine retriever (baseline RAG)
│   ├── rag_pipeline.py        # Vanilla RAG eval over multiple top_k
│   ├── finetuning.py          # Q-LoRA training + adapter loading
│   └── quasar/
│       ├── triples.py         # Triple loading + QID enrichment
│       ├── embeddings.py      # Triple / relation / entity embedders
│       ├── index.py           # FAISS IndexFlatIP build / load
│       ├── retrieval.py       # Enriched query encoding + parallel FAISS
│       ├── parallel_generation.py
│       └── pipeline.py        # End-to-end QUASAR evaluation
├── scripts/
│   ├── explore_data.py        # Dataset EDA + figures
│   ├── run_parametric.py      # LLM-only baseline
│   ├── run_rag.py             # Vanilla RAG sweep over top_k
│   ├── build_quasar_index.py  # Pre-compute QUASAR resources
│   ├── run_quasar.py          # Full QUASAR evaluation
│   ├── run_finetune.py        # Q-LoRA fine-tuning
│   ├── eval_finetuned.py      # Evaluate a saved adapter
│   └── make_submission.py     # Kaggle-format submission CSV
├── data/                      # Place dataset here (see below)
│   └── cache/                 # Auto-generated caches (not in git)
├── requirements.txt
└── README.md
```

## Setup

1. Clone and create a virtual environment (Python 3.10+ recommended).
   ```bash
   git clone https://github.com/<your-handle>/quasar-temporal-qa
   cd quasar-temporal-qa
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Authenticate with Hugging Face so the gated Llama-3.2-3B-Instruct weights can be downloaded.
   ```bash
   huggingface-cli login
   ```
3. Place the dataset files under `data/`:
   ```
   data/
     train.csv
     val.csv
     test.csv
     facts.json
     label_mappings.json
     sample_submission.csv
   ```
   You can override the location with the `DATA_PATH` environment variable.

A CUDA-enabled GPU is required for 4-bit quantization. The original work targeted a Colab T4 (16 GB).

## Running the pipelines

All scripts are idempotent — heavy artifacts (embeddings, FAISS index, model responses) are cached under `data/cache/` and re-used on subsequent runs.

### 1. Explore the data
```bash
python scripts/explore_data.py
```
Saves question-type and answer-length plots under `docs/figures/`.

### 2. Parametric-memory baseline
```bash
python scripts/run_parametric.py --prompt-type standard_nl --num-shots 0
python scripts/run_parametric.py --prompt-type standard_nl --num-shots 5
python scripts/run_parametric.py --prompt-type standard_id --num-shots 5
```

### 3. Vanilla RAG (MiniLM over flat facts)
```bash
python scripts/run_rag.py --top-k 1 3 5 10 --prompt-type standard_id --num-shots 10
```
Produces a top_k comparison table and a saved figure under `docs/figures/`.

### 4. QUASAR (FAISS + enriched embeddings)
```bash
# (optional) build the index in advance
python scripts/build_quasar_index.py

# run the full QUASAR evaluation on the validation split
python scripts/run_quasar.py --top-k 5 --split val --prompt-type standard_id
```

### 5. Q-LoRA fine-tuning
```bash
# Fine-tune for natural-language answers
python scripts/run_finetune.py --prompt-type standard_nl --response-type nl

# Fine-tune for entity-ID answers
python scripts/run_finetune.py --prompt-type standard_id --response-type id

# Evaluate a saved adapter
python scripts/eval_finetuned.py \
    --adapter-path data/cache/qlora-natural_lang_answers \
    --prompt-type standard_nl --response-type nl
```

### 6. Generate a Kaggle submission
```bash
python scripts/make_submission.py --top-k 5 --output submission.csv
```

## Evaluation

All systems are evaluated with an Exact-Match (EM) and Partial-Match (PM) F1 metric implemented in `quasar_rag/evaluation.py`. Predictions and gold answers are normalized (lowercased + punctuation stripped) and matched as multi-set overlaps before averaging precision, recall and F1 across questions.

## Notes on reproducibility

- Embedding cache files (`relation_embeddings.npy`, `nlentity2emb.json`, `entity2emb.json`, `triple_embeddings.npy`, `faiss_triples.index`) are written next to the dataset.
- LLM response caches use deterministic filenames keyed by split, prompt type and top_k, so reruns of evaluation are instantaneous.
- The Llama generation uses `temperature=0.3, top_p=0.8` by default, which gives near-deterministic outputs but isn't strictly reproducible bit-for-bit.

## Acknowledgments

This codebase is a refactor of an INF8460 (Automne 2025) team project at Polytechnique Montréal for a Kaggle competition on temporal question answering. The base model is `meta-llama/Llama-3.2-3B-Instruct` and the embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
