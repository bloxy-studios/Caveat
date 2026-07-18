"""Per-drug predictor: logistic regression + isotonic calibration + Mondrian
conformal + evidence typing + target gate -> a DrugResult.

One model per antibiotic. Each model is self-describing enough to render a full
antibiotic-response card.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import __version__
from .calibration import apply_isotonic, fit_isotonic
from .config import CAVEAT_MESSAGE, SPECIES
from .conformal import MondrianBinaryConformal, set_to_call
from .drug_kb import (
    association_families,
    curated_families,
    drug_entry,
    mechanism_coverage,
    mechanism_note,
    molecular_target_str,
)
from .features import row_to_vector
from .schemas import DrugResult, GenomeReport, SupportingFeature, TargetGate
from .target_gate import apply_target_gate

EvidenceList = List[Dict[str, object]]


@dataclass
class DrugModel:
    drug: str
    display_name: str
    drug_class: str
    molecular_target: str
    feature_columns: List[str]
    lr: LogisticRegression
    iso: object
    conformal: MondrianBinaryConformal

    # ---- calibrated probability helpers ----
    def _p_raw(self, X: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(X)[:, 1]

    def predict_calibrated(self, X: np.ndarray) -> np.ndarray:
        return apply_isotonic(self.iso, self._p_raw(X))

    def coef_map(self) -> Dict[str, float]:
        return {c: float(w) for c, w in zip(self.feature_columns, self.lr.coef_[0])}

    # ---- single-genome inference ----
    def predict_one(
        self,
        feature_row: Dict[str, float],
        evidence_list: Optional[EvidenceList],
        alpha: float,
        kb: Optional[Dict] = None,
    ) -> DrugResult:
        x = np.array([row_to_vector(feature_row, self.feature_columns)], dtype=float)
        p_cal = float(self.predict_calibrated(x)[0])
        pset = self.conformal.predict_set(p_cal, alpha)
        call, reason = set_to_call(pset)

        # deterministic molecular-target gate (can only withhold "works")
        gate_raw = apply_target_gate(self.drug, feature_row, call, kb)
        if gate_raw["forced_no_call"] and call == "likely_to_work":
            call, reason = "no_call", "target_gate"
        gate = TargetGate(
            target_present=bool(gate_raw["target_present"]),
            mechanism_coverage=gate_raw["mechanism_coverage"],
            forced_no_call=bool(gate_raw["forced_no_call"]),
            note=str(gate_raw["note"]),
        )

        # annotation-quality guard: partial/low-identity hits -> no-call
        if feature_row.get("annq__any_partial", 0.0) >= 1.0 and call != "no_call":
            call, reason = "no_call", "poor_annotation_quality"

        evidence_type, supporting, summary = self._explain(feature_row, evidence_list, call, kb)

        if call == "likely_to_fail":
            confidence: Optional[float] = round(p_cal, 4)
        elif call == "likely_to_work":
            confidence = round(1.0 - p_cal, 4)
        else:
            confidence = None  # no directional call -> see p_resistant instead

        return DrugResult(
            drug=self.drug,
            display_name=self.display_name,
            drug_class=self.drug_class,
            molecular_target=self.molecular_target,
            call=call,
            no_call_reason=reason,
            confidence=confidence,
            p_resistant=round(p_cal, 4),
            prediction_set=pset,
            evidence_type=evidence_type,
            evidence_summary=summary,
            supporting_features=supporting,
            target_gate=gate,
            caveat=CAVEAT_MESSAGE,
        )

    def _explain(self, feature_row, evidence_list, call, kb):
        cur = curated_families(self.drug, kb)
        assoc = association_families(self.drug, kb)
        cur_present = [f for f in cur if feature_row.get(f"presence__{f}", 0.0) >= 1.0]
        assoc_present = [f for f in assoc if feature_row.get(f"presence__{f}", 0.0) >= 1.0]
        ev_by_fam = {}
        for ev in (evidence_list or []):
            ev_by_fam.setdefault(ev.get("gene"), ev)
        coefs = self.coef_map()

        supporting: List[SupportingFeature] = []
        for fam in cur_present:
            ev = ev_by_fam.get(fam, {})
            supporting.append(
                SupportingFeature(
                    gene=fam,
                    detail=mechanism_note(self.drug, fam, kb) or "curated resistance determinant",
                    method=ev.get("method"),
                    pct_identity=ev.get("pct_identity"),
                    pct_coverage=ev.get("pct_coverage"),
                    curated=True,
                    model_contribution=round(coefs.get(f"presence__{fam}", 0.0), 4),
                )
            )
        for fam in assoc_present:
            ev = ev_by_fam.get(fam, {})
            supporting.append(
                SupportingFeature(
                    gene=fam,
                    detail="statistically associated feature — NOT a curated cause for this drug",
                    method=ev.get("method"),
                    pct_identity=ev.get("pct_identity"),
                    pct_coverage=ev.get("pct_coverage"),
                    curated=False,
                    model_contribution=round(coefs.get(f"presence__{fam}", 0.0), 4),
                )
            )

        if cur_present:
            etype = "i"
            if call == "likely_to_work":
                summary = (
                    f"Curated determinant(s) present ({', '.join(cur_present)}), but the calibrated "
                    f"model still predicts susceptibility for {self.display_name} — interpret with "
                    f"caution and confirm by laboratory testing."
                )
            else:
                summary = (
                    f"Curated resistance determinant(s) detected: {', '.join(cur_present)}. "
                    f"This is a known mechanism for {self.display_name}."
                )
        elif assoc_present:
            etype = "ii"
            summary = (
                f"No curated resistance gene for {self.display_name}; decision leans on "
                f"associated feature(s): {', '.join(assoc_present)}. Association, not proof of cause."
            )
        else:
            etype = "iii"
            summary = f"No known resistance signal for {self.display_name} in the detected features."
        return etype, supporting, summary


def fit_drug_model(
    drug: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_iso: np.ndarray,
    y_iso: np.ndarray,
    X_conf: np.ndarray,
    y_conf: np.ndarray,
    feature_columns: List[str],
    kb: Optional[Dict] = None,
) -> DrugModel:
    """Fit LR on train, isotonic on one calibration slice, conformal on another."""
    entry = drug_entry(drug, kb)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(X_train, y_train)

    iso = fit_isotonic(lr.predict_proba(X_iso)[:, 1], y_iso)
    p_conf_cal = apply_isotonic(iso, lr.predict_proba(X_conf)[:, 1])
    conf = MondrianBinaryConformal().fit(p_conf_cal, y_conf)

    return DrugModel(
        drug=drug,
        display_name=entry.get("display_name", drug),
        drug_class=entry.get("drug_class", ""),
        molecular_target=molecular_target_str(drug, kb),
        feature_columns=list(feature_columns),
        lr=lr,
        iso=iso,
        conformal=conf,
    )


def build_report(
    genome_id: str,
    feature_row: Dict[str, float],
    evidence_list: Optional[EvidenceList],
    models: Dict[str, DrugModel],
    alpha: float,
    kb: Optional[Dict] = None,
) -> GenomeReport:
    results = [m.predict_one(feature_row, evidence_list, alpha, kb) for m in models.values()]
    return GenomeReport(
        genome_id=genome_id,
        species=SPECIES,
        annotator="AMRFinderPlus",
        model_version=__version__,
        alpha=alpha,
        generated_at=datetime.now(timezone.utc).isoformat(),
        drug_results=results,
        caveat=CAVEAT_MESSAGE,
        disclaimers=[
            "Defensive decision support only — never a treatment decision.",
            "Performance is reported on a genetically grouped hold-out split.",
            "Evidence type i = curated resistance gene; type ii = statistical association (not proof of cause).",
        ],
    )
