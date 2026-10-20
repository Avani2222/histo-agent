"""LLM-as-judge: does each cited abstract actually support the claim attached to it?

Judges are noisy, so spot-check ~30 judgements by hand and report agreement in the README.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

JUDGE_SYSTEM = ("You check citations. Given a claim and a PubMed abstract, answer whether the abstract "
                "supports the claim. Be strict: partial or tangential support counts as not supported.")


class Verdict(BaseModel):
    supported: bool
    reason: str


def judge_citation(llm, claim: str, abstract: str) -> Verdict:
    msgs = [SystemMessage(JUDGE_SYSTEM),
            HumanMessage(f"Claim: {claim}\n\nAbstract: {abstract[:2000]}\n\nIs the claim supported?")]
    return llm.with_structured_output(Verdict).invoke(msgs)


def faithfulness(llm, report: dict, hits_by_pmid: dict) -> float | None:
    """Share of citations whose abstract supports the stated claim. None if there are no citations."""
    cites = report.get("evidence", [])
    if not cites:
        return None
    ok = sum(judge_citation(llm, c["supports"], hits_by_pmid[c["pmid"]]["abstract"]).supported
             for c in cites if c["pmid"] in hits_by_pmid)
    return ok / len(cites)
