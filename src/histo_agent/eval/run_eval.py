"""Evaluation harness: run the full agent on held-out test images and measure it.

Usage: python -m histo_agent.eval.run_eval --n 150 [--judge]

Metrics
- vision sensitivity / specificity on cases the agent did not refer
- refusal rate (share sent to "uncertain / refer to pathologist") and how many refusals were actually errors
- citation validity: share of LLM citations that pointed at a retrieved PMID (before the verify step removed bad ones)
- faithfulness (with --judge): LLM-judge share of citations whose abstract supports the claim
- latency p50 / p95
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from histo_agent.agent.graph import build_default_graph, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--splits", default="data/processed/splits.csv")
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--judge", action="store_true", help="also run the LLM citation judge (costs tokens)")
    ap.add_argument("--out", default="artifacts/eval")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    df = pd.read_csv(a.splits)
    test = df[df.split == "test"]
    # stratified sample so both classes are represented
    sample = (test.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), a.n // 2), random_state=a.seed)))

    graph = build_default_graph(a.config)
    judge_llm = None
    if a.judge:
        from histo_agent.agent.llm import get_llm
        judge_llm = get_llm()
        from histo_agent.eval.judges import faithfulness

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for _, r in sample.iterrows():
            res = run(r.path, graph)
            rep = res["report"]
            row = {"path": r.path, "patient_id": r.patient_id, "label": int(r.label), "finding": rep["finding"],
                   "p": rep["model_probability_malignant"], "flags": res["flags"],
                   "n_citations": len(rep["evidence"]), "total_s": res["timings"]["total_s"]}
            removed = sum(int(fl.split("_")[1]) for fl in res["flags"] if fl.startswith("removed_"))
            row["n_citations_proposed"] = row["n_citations"] + removed
            if judge_llm and rep["evidence"]:
                hits = {h["pmid"]: h for h in res["hits"]}
                row["faithfulness"] = faithfulness(judge_llm, rep, hits)
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    res = pd.DataFrame(rows)
    decided = res[res.finding != "uncertain"]
    pred = (decided.finding == "malignant").astype(int)
    tp = int(((pred == 1) & (decided.label == 1)).sum())
    fn = int(((pred == 0) & (decided.label == 1)).sum())
    tn = int(((pred == 0) & (decided.label == 0)).sum())
    fp = int(((pred == 1) & (decided.label == 0)).sum())
    proposed = res.n_citations_proposed.sum()
    summary = {
        "n_cases": len(res),
        "refusal_rate": float((res.finding == "uncertain").mean()),
        "sensitivity_on_decided": tp / max(tp + fn, 1),
        "specificity_on_decided": tn / max(tn + fp, 1),
        "citation_validity": float(res.n_citations.sum() / proposed) if proposed else None,
        "faithfulness_mean": float(res.faithfulness.dropna().mean()) if "faithfulness" in res else None,
        "latency_p50_s": float(np.percentile(res.total_s, 50)),
        "latency_p95_s": float(np.percentile(res.total_s, 95)),
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
