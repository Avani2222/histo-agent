# HistoAgent: project scope

## Problem
Deep learning models for histopathology give a probability, not an explanation, and they are rarely honest about
uncertainty. LLM assistants explain fluently but invent facts and citations. This project combines the two
carefully: a vision model makes the call, an agent grounds the explanation in real literature, and the system
refuses when the model is unsure. Every claim is measured.

## Goal
Given a single breast histopathology image patch, produce a structured, cited, uncertainty-aware report, and
quantify how often each part of the system is wrong.

## In scope
| Area | Scope |
|---|---|
| Data | BreakHis (82 patients, 7,909 images), 400X magnification first; patient-level train/val/test splits |
| Vision | Binary benign vs malignant classifier (timm EfficientNet-B0 baseline); threshold tuned on validation for high sensitivity; Grad-CAM explanations |
| Evidence | 300-500 PubMed abstracts on breast histopathology, embedded with bge-small and searched with FAISS |
| Agent | LangGraph flow: classify, refer when uncertain, retrieve, draft a structured report (Pydantic), verify that every citation was actually retrieved |
| Evaluation | 150 held-out test cases: sensitivity, specificity, refusal rate, citation validity, LLM-judge faithfulness (spot-checked by hand), p50/p95 latency, cost per report |
| Engineering | FastAPI endpoint, Streamlit demo, Docker image, GitHub Actions CI (lint, tests, Docker build), MLflow experiment tracking |
| Deployment | One public cloud deployment (AWS App Runner or ECS preferred; Google Cloud Run as fallback) |

## Out of scope
- Clinical use, diagnosis or any medical claim. This is a research and portfolio project.
- Whole-slide image (WSI) processing; the input is a single patch.
- Subtype classification (8 BreakHis subtypes) in the first version; a possible extension.
- Fine-tuning the LLM; the agent uses an off-the-shelf model through an OpenAI-compatible API.
- User accounts, storage of uploaded images, PHI handling.

## Success criteria (definition of done)
1. Patient-level test AUROC reported honestly, with the threshold chosen on validation only.
2. The agent refuses on low-confidence cases, and the README reports how many refusals were cases the model would have got wrong.
3. 100% of citations in final reports point to retrieved PMIDs (enforced by the verify step); judge faithfulness reported with hand-checked agreement.
4. `pytest` and CI green; Docker image builds; the deployed URL works for a reviewer.
5. README has the results table, architecture diagram, demo GIF and "how to run"; a 2-minute demo video exists.

## Deliverables
- Public GitHub repo `histo-agent` with code, tests, CI and README
- Trained model checkpoint and evaluation reports (`artifacts/vision/test_metrics.json`, `artifacts/eval/summary.json`)
- Live demo URL and a 2-minute demo video
- Resume bullet with real numbers (template in PLAN.md)

## Timeline
| Weekend | Outcome |
|---|---|
| 1 | Patient-level splits, trained classifier, test metrics, Grad-CAM examples |
| 2 | PubMed index, working LangGraph agent, tuned uncertainty band |
| 3 | 150-case evaluation, API + UI, Docker, cloud deployment, README and demo |

## Possible extensions (after v1)
- 8-class subtype prediction and multi-magnification fusion
- Stain normalisation (Macenko) and a cross-dataset test (e.g. BACH) to measure generalisation
- Swap the CNN for a pathology foundation model backbone and compare
- Human-in-the-loop review step in the agent graph (LangGraph interrupt)
- Drift monitoring on incoming images

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Small GPU (MX550, 2 GB) makes training slow | Small model and batch size by default; free Kaggle or Colab T4 for full runs |
| Only 82 patients, so metrics are noisy | Report patient-level results and the patient counts per split; avoid over-claiming |
| LLM judge is unreliable | Hand-check 30 judgements and report agreement |
| Cloud cost | Free tiers; CPU-only image; scale to zero when idle |
