"""Fast tests that need no GPU, data or API key: split parsing, confidence bands, and the agent graph with fakes."""
import importlib.util
from pathlib import Path

from histo_agent.agent.graph import build_graph, confidence_band
from histo_agent.agent.schemas import Citation, DraftReport

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("splits", ROOT / "scripts" / "make_patient_splits.py")
splits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splits)

CFG = {"uncertainty_band": 0.15, "retrieval": {"top_k": 3}}


def test_parse_breakhis_filename():
    row = splits.parse(Path("x/SOB_M_DC-14-2523-400-012.png"))
    assert row["label"] == 1 and row["patient_id"] == "14-2523"
    assert row["subtype"] == "ductal_carcinoma" and row["magnification"] == "400X"


def test_confidence_band():
    assert confidence_band(0.52, 0.5, 0.15) == "low"
    assert confidence_band(0.75, 0.5, 0.15) == "medium"
    assert confidence_band(0.95, 0.5, 0.15) == "high"


class FakePredictor:
    def __init__(self, p):
        self.p = p

    def predict(self, _):
        return {"p_malignant": self.p, "threshold": 0.5, "label": "malignant" if self.p >= 0.5 else "benign"}


class FakeRetriever:
    def search(self, query, top_k=5):
        return [{"pmid": str(i), "title": f"t{i}", "abstract": "a", "year": "2020", "journal": "j", "score": 0.9}
                for i in range(top_k)]


class FakeLLM:
    """Mimics llm.with_structured_output(DraftReport).invoke(...)."""
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _msgs):
        return DraftReport(summary="s", limitations=["l"], recommendation="r",
                           evidence=[Citation(pmid="0", supports="x"), Citation(pmid="999", supports="made up")])


def test_low_confidence_is_referred_without_llm():
    g = build_graph(FakePredictor(0.51), FakeRetriever(), FakeLLM(), CFG)
    out = g.invoke({"image_path": "x.png"})
    assert out["report"]["finding"] == "uncertain"
    assert "refused_low_confidence" in out["flags"]


def test_unsupported_citations_are_removed():
    g = build_graph(FakePredictor(0.97), FakeRetriever(), FakeLLM(), CFG)
    out = g.invoke({"image_path": "x.png"})
    pmids = [c["pmid"] for c in out["report"]["evidence"]]
    assert pmids == ["0"]
    assert any(f.startswith("removed_1_") for f in out["flags"])
