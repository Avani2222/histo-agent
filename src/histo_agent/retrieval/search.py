"""Query-time retrieval over the FAISS index."""
import json
from functools import lru_cache
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

# bge models expect this prefix on queries (not on documents)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self, index_dir: str):
        d = Path(index_dir)
        info = json.load(open(d / "info.json"))
        self.model = SentenceTransformer(info["embed_model"])
        self.index = faiss.read_index(str(d / "pubmed.faiss"))
        with open(d / "meta.jsonl", encoding="utf-8") as f:
            self.meta = [json.loads(line) for line in f]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(q, top_k)
        hits = []
        for s, i in zip(scores[0], idx[0], strict=True):
            if i < 0:
                continue
            doc = self.meta[i]
            hits.append({"pmid": doc["pmid"], "title": doc["title"], "abstract": doc["abstract"],
                         "year": doc.get("year", ""), "journal": doc.get("journal", ""), "score": float(s)})
        return hits


@lru_cache(maxsize=1)
def get_retriever(index_dir: str) -> Retriever:
    return Retriever(index_dir)
