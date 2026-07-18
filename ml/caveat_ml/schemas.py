"""Pydantic models = the single source of truth for the /predict contract.

The Next.js UI and the Gradio wrapper both consume `GenomeReport`. Every result
carries a calibrated confidence, an explicit evidence type, and the mandatory
lab-confirmation caveat.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Literal

CallType = Literal["likely_to_fail", "likely_to_work", "no_call"]
EvidenceType = Literal["i", "ii", "iii"]
NoCallReason = Literal[
    "conflicting_evidence",     # conformal set = {R,S}
    "unlike_training_data",     # conformal set = {} (out-of-distribution)
    "low_confidence",           # calibrated prob in the uncertain band
    "target_gate",              # cannot interrogate the drug's mechanism -> won't claim "works"
    "poor_annotation_quality",  # partial/low-identity hits
]
MechanismCoverage = Literal["full", "partial", "none"]


class SupportingFeature(BaseModel):
    gene: str = Field(..., description="Canonical gene family, e.g. blaKPC")
    detail: str = Field("", description="Human-readable mechanism note")
    method: Optional[str] = Field(None, description="AMRFinderPlus method (EXACTX/BLASTX/HMM/POINTX)")
    pct_identity: Optional[float] = None
    pct_coverage: Optional[float] = None
    curated: bool = Field(
        False,
        description="True = curated resistance determinant for THIS drug (evidence type i). "
        "False = statistical association only (evidence type ii).",
    )
    model_contribution: Optional[float] = Field(
        None, description="Signed logistic-regression contribution (coef * value). Association, not proof of cause."
    )


class TargetGate(BaseModel):
    target_present: bool = True
    mechanism_coverage: MechanismCoverage = "full"
    forced_no_call: bool = False
    note: str = ""


class DrugResult(BaseModel):
    drug: str
    display_name: str
    drug_class: str
    molecular_target: str
    call: CallType
    no_call_reason: Optional[NoCallReason] = None
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Calibrated confidence in the reported directional call. None for no-call (see p_resistant).",
    )
    p_resistant: float = Field(..., ge=0.0, le=1.0, description="Calibrated P(resistant)")
    prediction_set: List[str] = Field(..., description="Conformal set, subset of ['R','S']")
    evidence_type: EvidenceType
    evidence_summary: str = ""
    supporting_features: List[SupportingFeature] = Field(default_factory=list)
    target_gate: TargetGate = Field(default_factory=TargetGate)
    caveat: str


class GenomeReport(BaseModel):
    genome_id: str
    species: str
    annotator: str = "AMRFinderPlus"
    model_version: str
    alpha: float
    generated_at: str
    drug_results: List[DrugResult]
    caveat: str
    disclaimers: List[str] = Field(default_factory=list)


class PredictRequest(BaseModel):
    genome_id: Optional[str] = Field(None, description="Look up precomputed features for a held-out genome")
    features: Optional[Dict[str, float]] = Field(None, description="Direct feature vector (overrides genome_id)")
    alpha: Optional[float] = Field(None, ge=0.01, le=0.5, description="Conformal significance level")
