from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    pmid: str = Field(description="PubMed ID of a retrieved abstract")
    supports: str = Field(description="One sentence: what this source supports in the report")


class Report(BaseModel):
    """Structured output the agent must produce. Validated before it is returned."""
    finding: Literal["benign", "malignant", "uncertain"]
    model_probability_malignant: float = Field(ge=0, le=1)
    confidence: Literal["high", "medium", "low"]
    summary: str = Field(description="2-4 sentences, plain language, no diagnosis beyond the model output")
    evidence: list[Citation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendation: str
    disclaimer: str = "Research prototype. Not for clinical use; a pathologist must review every case."


class DraftReport(BaseModel):
    """What the LLM is asked to fill in; deterministic fields are added by the graph."""
    summary: str
    evidence: list[Citation]
    limitations: list[str]
    recommendation: str
