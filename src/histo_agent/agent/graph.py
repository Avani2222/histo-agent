"""LangGraph agent: classify -> (uncertain? refer) -> retrieve -> draft -> verify citations.

Usage: python -m histo_agent.agent.graph --image path/to/patch.png
"""
import argparse
import json
import time
from typing import Any, TypedDict

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from histo_agent.agent.prompts import SYSTEM, USER_TEMPLATE, format_evidence
from histo_agent.agent.schemas import DraftReport, Report


class AgentState(TypedDict, total=False):
    image_path: str
    vision: dict            # {"p_malignant", "threshold", "label"}
    confidence: str         # high | medium | low
    hits: list[dict]        # retrieved abstracts
    draft: dict             # DraftReport fields from the LLM
    report: dict            # final validated Report
    flags: list[str]        # problems found during verification
    timings: dict


def confidence_band(p: float, thr: float, band: float) -> str:
    margin = abs(p - thr)
    if margin < band:
        return "low"
    return "high" if margin >= 2 * band else "medium"


def build_graph(predictor, retriever, llm, cfg: dict):
    """Dependencies are injected so tests can pass fakes."""
    band, top_k = cfg["uncertainty_band"], cfg["retrieval"]["top_k"]

    def classify(state: AgentState) -> AgentState:
        t = time.perf_counter()
        v = predictor.predict(state["image_path"])
        return {"vision": v, "confidence": confidence_band(v["p_malignant"], v["threshold"], band),
                "timings": {"classify_s": time.perf_counter() - t}}

    def route(state: AgentState) -> str:
        return "refer" if state["confidence"] == "low" else "retrieve"

    def refer(state: AgentState) -> AgentState:
        v = state["vision"]
        report = Report(
            finding="uncertain", model_probability_malignant=v["p_malignant"], confidence="low",
            summary=("The model's output is too close to its decision threshold to support a benign or "
                     "malignant call for this patch."),
            limitations=["Model probability lies inside the uncertainty band around the threshold."],
            recommendation="Refer to a pathologist for review; consider additional patches or magnifications.",
        )
        return {"report": report.model_dump(), "flags": ["refused_low_confidence"]}

    def retrieve(state: AgentState) -> AgentState:
        t = time.perf_counter()
        label = state["vision"]["label"]
        query = f"{label} breast lesion histopathology features diagnosis"
        hits = retriever.search(query, top_k=top_k)
        return {"hits": hits, "timings": {**state.get("timings", {}), "retrieve_s": time.perf_counter() - t}}

    def draft(state: AgentState) -> AgentState:
        t = time.perf_counter()
        v = state["vision"]
        msgs = [SystemMessage(SYSTEM), HumanMessage(USER_TEMPLATE.format(
            label=v["label"], p=v["p_malignant"], thr=v["threshold"], confidence=state["confidence"],
            evidence=format_evidence(state["hits"])))]
        out: DraftReport = llm.with_structured_output(DraftReport).invoke(msgs)
        return {"draft": out.model_dump(), "timings": {**state.get("timings", {}), "draft_s": time.perf_counter() - t}}

    def verify(state: AgentState) -> AgentState:
        allowed = {h["pmid"] for h in state["hits"]}
        d, flags = state["draft"], []
        kept = [c for c in d["evidence"] if c["pmid"] in allowed]
        if len(kept) < len(d["evidence"]):
            flags.append(f"removed_{len(d['evidence']) - len(kept)}_unsupported_citations")
        if not kept:
            flags.append("no_supported_citations")
        v = state["vision"]
        report = Report(finding=v["label"], model_probability_malignant=v["p_malignant"],
                        confidence=state["confidence"], summary=d["summary"], evidence=kept,
                        limitations=d["limitations"], recommendation=d["recommendation"])
        return {"report": report.model_dump(), "flags": flags}

    g = StateGraph(AgentState)
    for name, fn in [("classify", classify), ("refer", refer), ("retrieve", retrieve),
                     ("draft", draft), ("verify", verify)]:
        g.add_node(name, fn)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route, {"refer": "refer", "retrieve": "retrieve"})
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "verify")
    g.add_edge("refer", END)
    g.add_edge("verify", END)
    return g.compile()


def build_default_graph(config_path: str = "configs/agent.yaml"):
    from histo_agent.agent.llm import get_llm
    from histo_agent.retrieval.search import get_retriever
    from histo_agent.vision.predictor import get_predictor

    cfg = yaml.safe_load(open(config_path))
    return build_graph(
        get_predictor(cfg["vision_ckpt"], cfg["threshold_file"]),
        get_retriever(cfg["retrieval"]["index_dir"]),
        get_llm(cfg["llm"]["temperature"], cfg["llm"]["max_tokens"]),
        cfg,
    )


def run(image_path: str, graph=None) -> dict[str, Any]:
    graph = graph or build_default_graph()
    t = time.perf_counter()
    state = graph.invoke({"image_path": image_path})
    return {"report": state["report"], "flags": state.get("flags", []),
            "retrieved_pmids": [h["pmid"] for h in state.get("hits", [])],
            "hits": state.get("hits", []),
            "timings": {**state.get("timings", {}), "total_s": time.perf_counter() - t}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    print(json.dumps(run(ap.parse_args().image), indent=2))
