# HistoAgent build plan

Tick items as you go. Each weekend ends with something you can show.

## Setup (30 min)
- [ ] Create a GitHub repo `histo-agent` (public) and push this folder.
- [ ] `python -m venv .venv`, activate it, `pip install -r requirements.txt`, `pip install -e .`
- [ ] `pytest -q` passes (tests use fakes, no data needed).
- [ ] Copy `.env.example` to `.env`, add an LLM key (OpenRouter or OpenAI; gpt-4o-mini costs cents for this whole project).

## Weekend 1: vision model
- [ ] Download BreakHis (see `scripts/download_breakhis.md`) into `data/raw/BreaKHis_v1`.
- [ ] `python scripts/make_patient_splits.py --root data/raw/BreaKHis_v1 --mag 400X`
      and note the train/val/test patient counts.
      train 58 patients (17 B / 41 M), val 12 (4 B / 8 M), test 11 (3 B / 8 M); 1,820 images at 400X, seed 42

- [ ] Train: `python -m histo_agent.vision.train`
  - On your laptop (MX550, 2 GB): keep `efficientnet_b0`, `batch_size: 16`, `img_size: 224`.
  - Faster: a free Kaggle or Colab GPU (T4). Upload `data/processed/splits.csv` + images, or use the Kaggle BreakHis dataset directly.
- [ ] `mlflow ui` and screenshot the val AUROC curve for the README.
- [ ] Evaluate: `python -m histo_agent.vision.evaluate` gives you `artifacts/vision/test_metrics.json`.
- [ ] Grad-CAM on 3-4 test images: `python -m histo_agent.vision.gradcam --image ...`
- [ ] **Resume milestone:** you can now add the project as "in progress" with the real patient-level test AUROC.

## Weekend 2: retrieval + agent
- [ ] `python scripts/fetch_pubmed.py --max 400`
- [ ] `python -m histo_agent.retrieval.index`
- [ ] Sanity-check retrieval: search 3 queries by hand and read the top abstracts. Are they relevant?
- [ ] `python -m histo_agent.agent.graph --image <a test image>`: read the report critically.
- [ ] Tune `uncertainty_band` in `configs/agent.yaml` so refusals catch most model errors without refusing everything.
- [ ] Export the graph diagram (`graph.get_graph().draw_mermaid()`) into the README.

## Weekend 3: evaluation + deployment
- [ ] `python -m histo_agent.eval.run_eval --n 150 --judge`
- [ ] Hand-check 30 citation judgements; report judge agreement in the README.
- [ ] Fill in the README Results table with the real numbers.
- [ ] `uvicorn histo_agent.api.app:app` + `streamlit run ui/app.py`, record a 2-minute demo video and a GIF.
- [ ] Docker build, then deploy:
  - AWS: push the image to ECR and run it on App Runner or ECS Fargate (copy `artifacts/` and `data/index/` into the image or an S3-backed volume).
  - Or Google Cloud Run (same as AirSight) if AWS is slow to set up; AWS is better for the "AWS experience" screening questions.
- [ ] GitHub Actions badge in the README.

## Resume bullet (fill with your real numbers)
> Built **HistoAgent**, a LangGraph agent that combines a fine-tuned PyTorch histopathology classifier
> (patient-level AUROC 0.xx) with FAISS retrieval over PubMed to generate cited, uncertainty-aware reports.
> Evaluated on a 150-case harness (citation validity xx%, judge faithfulness xx%, sensitivity xx%,
> refusal rate xx%); deployed on AWS with Docker and GitHub Actions CI (p95 x.x s/report).

## Interview questions to be ready for
- Why patient-level splits? What happens to AUROC with random splits? (Try it once and report the gap.)
- How did you pick the threshold, and why optimise sensitivity?
- How do you know a citation is real and relevant? What does the judge miss?
- When does the agent refuse, and how often is a refusal a case the model would have got wrong?
- Cost and latency per report; what would you change for 10,000 reports a day?
