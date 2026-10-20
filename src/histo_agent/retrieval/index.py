"""Build a FAISS index over PubMed abstracts.

Usage: python -m histo_agent.retrieval.index --docs data/processed/pubmed.jsonl --out data/index
Each abstract is one chunk (abstracts are short); title is prepended so it is searchable.
"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def load_docs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/processed/pubmed.jsonl")
    ap.add_argument("--out", default="data/index")
    a = ap.parse_args()

    docs = load_docs(a.docs)
    texts = [f"{d['title']}\n{d['abstract']}" for d in docs]
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # inner product on normalized vectors = cosine similarity
    index.add(emb)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "pubmed.faiss"))
    with open(out / "meta.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    json.dump({"embed_model": EMBED_MODEL, "n_docs": len(docs)}, open(out / "info.json", "w"))
    print(f"indexed {len(docs)} abstracts -> {out}")


if __name__ == "__main__":
    main()
