"""Fetch PubMed abstracts on breast histopathology via NCBI E-utilities (free, no key needed for small volumes)."""
import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from dotenv import load_dotenv

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_QUERIES = [
    "breast carcinoma histopathology diagnosis",
    "invasive ductal carcinoma histological features",
    "lobular carcinoma histopathology",
    "fibroadenoma histopathology",
    "phyllodes tumor histological classification",
    "mucinous carcinoma breast histology",
    "papillary carcinoma breast histopathology",
    "adenosis breast benign lesion histology",
    "deep learning breast histopathology image classification",
    "Nottingham histological grade breast cancer",
]


def search(term, retmax, email):
    r = requests.get(f"{EUTILS}/esearch.fcgi", params={
        "db": "pubmed", "term": term + " AND hasabstract", "retmax": retmax, "retmode": "json", "email": email},
        timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def fetch(pmids, email):
    r = requests.get(f"{EUTILS}/efetch.fcgi", params={
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "email": email}, timeout=60)
    r.raise_for_status()
    out = []
    for art in ET.fromstring(r.content).findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID")
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        abstract = " ".join("".join(a.itertext()) for a in art.findall(".//AbstractText"))
        year = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""
        journal = art.findtext(".//Journal/Title") or ""
        if abstract:
            out.append({"pmid": pmid, "title": title, "abstract": abstract, "year": year[:4], "journal": journal})
    return out


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=400, help="approximate total abstracts")
    ap.add_argument("--out", default="data/processed/pubmed.jsonl")
    args = ap.parse_args()
    email = os.getenv("NCBI_EMAIL", "anonymous@example.com")

    per_query = max(10, args.max // len(DEFAULT_QUERIES))
    seen, docs = set(), []
    for q in DEFAULT_QUERIES:
        ids = [i for i in search(q, per_query, email) if i not in seen]
        seen.update(ids)
        for k in range(0, len(ids), 100):
            docs += fetch(ids[k:k + 100], email)
            time.sleep(0.4)  # stay under NCBI's 3 requests/second
        print(f"{q!r}: total {len(docs)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} abstracts to {args.out}")


if __name__ == "__main__":
    main()
