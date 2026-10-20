"""FastAPI service. Run: uvicorn histo_agent.api.app:app --reload"""
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from histo_agent.agent.graph import build_default_graph, run

app = FastAPI(title="HistoAgent", version="0.1.0",
              description="Research prototype. Not for clinical use.")
_graph = None


def graph():
    global _graph
    if _graph is None:
        _graph = build_default_graph()
    return _graph


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg", "image/tiff"}:
        raise HTTPException(415, "Upload a PNG, JPEG or TIFF image patch")
    suffix = Path(file.filename or "x.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        result = run(path, graph())
    finally:
        Path(path).unlink(missing_ok=True)
    result.pop("hits", None)  # keep the response small; PMIDs are listed in retrieved_pmids
    return result
