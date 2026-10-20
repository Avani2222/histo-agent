SYSTEM = """You write short, careful research reports about a breast histopathology image patch.
A vision model has already classified the patch. You do NOT diagnose; you explain the model output
and connect it to published evidence.

Rules:
- Use ONLY the evidence passages provided. Cite them by their PMID.
- Never cite a PMID that is not in the evidence list.
- If the evidence does not support a statement, leave the statement out.
- State limitations honestly (single patch, single magnification, model error rates, no clinical context).
- Plain language, 2-4 sentence summary."""

USER_TEMPLATE = """Model output:
- predicted class: {label}
- probability of malignancy: {p:.3f} (decision threshold {thr:.3f})
- confidence band: {confidence}

Evidence passages:
{evidence}

Write the report fields: summary, evidence (pmid + what it supports), limitations, recommendation."""


def format_evidence(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[PMID {h['pmid']}] {h['title']} ({h.get('journal', '')}, {h.get('year', '')})\n{h['abstract'][:1200]}"
        for h in hits
    )
