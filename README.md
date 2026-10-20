# HistoAgent

**An uncertainty-aware agent for breast histopathology: a fine-tuned vision model + retrieval over PubMed evidence + a LangGraph agent that writes structured, cited reports, with an evaluation harness that measures how often it is wrong.**

> Research / portfolio project. **Not a medical device and not for clinical use.**
>
> Scope, success criteria and timeline: [SCOPE.md](SCOPE.md) · Build checklist: [PLAN.md](PLAN.md)

## What it does

1. **Classify**: a PyTorch CNN (timm, EfficientNet-B0 by default) predicts benign vs malignant on a BreakHis image patch, with Grad-CAM for explainability.
2. **Decide**: if the model is not confident, the agent says so and recommends pathologist review instead of guessing.
3. **Retrieve**: FAISS search over PubMed abstracts on breast histopathology.
4. **Report**: an LLM drafts a structured report (finding, confidence, evidence with PMID citations, limitations), validated with Pydantic.
5. **Verify**: every cited PMID must come from the retrieved evidence; unsupported citations are removed and flagged.

```
image ──► classify ──► confident? ──no──► uncertain report (refer to pathologist)
                          │yes
                          ▼
                       retrieve (FAISS over PubMed) ──► draft report (LLM) ──► verify citations ──► report
```

## Results

> Fill these in from `python -m histo_agent.eval.run_eval` once you have run it. Report honest numbers.

| Metric | Value |
|---|---|
| Vision model, patient-level test AUROC | _tbd_ |
| Sensitivity / specificity at chosen threshold | _tbd_ |
| Citation validity (cited PMIDs that were actually retrieved) | _tbd_ |
| Faithfulness (LLM-judge, spot-checked by hand on 30 cases) | _tbd_ |
| Refusal rate on low-confidence cases | _tbd_ |
| p95 latency / cost per report | _tbd_ |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
pip install -e .

# 1. data (see scripts/download_breakhis.md), then:
python scripts/make_patient_splits.py --root data/raw/BreaKHis_v1 --mag 400X

# 2. train + evaluate the vision model
python -m histo_agent.vision.train --config configs/vision.yaml
python -m histo_agent.vision.evaluate --config configs/vision.yaml

# 3. build the evidence index
python scripts/fetch_pubmed.py --max 400
python -m histo_agent.retrieval.index

# 4. run the agent on one image
copy .env.example .env           # add your LLM API key
python -m histo_agent.agent.graph --image path\to\image.png

# 5. evaluation harness
python -m histo_agent.eval.run_eval --n 150

# 6. API + UI
uvicorn histo_agent.api.app:app --reload
streamlit run ui/app.py
```

## Design decisions (good interview material)

- **Patient-level splits.** BreakHis has many images per patient. Random image splits leak the same patient into train and test and inflate scores, so splits are done by patient.
- **Threshold chosen on validation, reported on test.** In a screening setting a missed malignancy costs more than a false alarm, so the threshold targets high sensitivity.
- **Refuse when uncertain.** Below a confidence band the agent does not produce a diagnosis-style answer.
- **Citations are verified, not trusted.** The LLM may only cite PMIDs it was shown.
- **Measured, not assumed.** Every claim in the Results table comes from `run_eval`.

## Stack

PyTorch, timm, scikit-learn, Grad-CAM · sentence-transformers, FAISS · LangGraph, LangChain, Pydantic · FastAPI, Streamlit · Docker, GitHub Actions · MLflow (experiment tracking)

## Data

- **BreakHis** (Spanhol et al., 2016): 7,909 breast tumour images from 82 patients, benign/malignant, 4 magnifications.
- **PubMed abstracts** fetched through NCBI E-utilities.
