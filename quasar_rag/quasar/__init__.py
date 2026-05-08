"""QUASAR – the team's competition entry.

QUASAR (Question-Answering with Structured And Retrieval-augmented embeddings)
indexes temporal triples (subject, relation, object, start_time, end_time) and
their Wikidata QIDs in a FAISS index, and retrieves with an enriched query
embedding that combines the question with its associated entities, NL entities
and temporal values.
"""
