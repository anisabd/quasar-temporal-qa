# Dataset directory

Place the temporal-QA files here:

```
data/
├── train.csv
├── val.csv
├── test.csv
├── facts.json
├── label_mappings.json
└── sample_submission.csv
```

`data/cache/` will be created automatically and used to store embeddings, the
FAISS index, model responses and saved adapters. It is git-ignored.
