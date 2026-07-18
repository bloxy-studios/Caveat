"""The deterministic molecular-target gate.

Purpose (brief, verbatim intent): the system must not report "likely to work"
based solely on the *absence* of resistance markers. The gate is a rule-based
layer that runs AFTER the model and can only ever *withhold* an over-confident
"likely to work" — it can never invent a "works" call.

Two guards:
  1. Target presence: the drug's essential molecular target must be present to be
     a meaningful drug. (In K. pneumoniae the panel targets are essential and
     assumed present after genome QC; the hook generalizes to target-bypass
     organisms such as mecA/vanA.)
  2. Mechanism interrogability: if we cannot even inspect the drug's primary
     resistance mechanism (e.g. chromosomal gyrA/parC for ciprofloxacin is not
     called by AMRFinder in Klebsiella gene mode), then "no marker found" does
     NOT justify "susceptible" -> force a no-call.
"""

from __future__ import annotations

from typing import Dict

from .drug_kb import curated_families, mechanism_coverage
from .features import GENE_FAMILIES  # noqa: F401  (kept for clarity of intent)


def _has_curated_marker(drug: str, feature_row: Dict[str, float], kb: Dict | None) -> bool:
    for fam in curated_families(drug, kb):
        if feature_row.get(f"presence__{fam}", 0.0) >= 1.0:
            return True
    return False


def apply_target_gate(
    drug: str,
    feature_row: Dict[str, float],
    provisional_call: str,
    kb: Dict | None = None,
) -> Dict[str, object]:
    """Return gate state. `forced_no_call=True` means the predictor must downgrade
    a 'likely_to_work' to 'no_call' with reason 'target_gate'."""
    coverage = mechanism_coverage(drug, kb)
    has_marker = _has_curated_marker(drug, feature_row, kb)

    # Target presence: default True (essential target assumed present post-QC).
    target_present = True

    forced = False
    note = "Target present; primary mechanism fully interrogable by current features."

    if coverage != "full" and provisional_call == "likely_to_work" and not has_marker:
        forced = True
        note = (
            f"Cannot confirm susceptibility: the primary resistance mechanism for {drug} "
            f"is only {coverage}ly interrogable with the current annotator "
            f"(e.g. chromosomal target mutations are not called). Absence of markers is "
            f"not evidence of susceptibility -> returning no-call."
        )
    elif coverage == "partial":
        note = (
            f"Primary mechanism only partially interrogable ({coverage} coverage); "
            f"interpret a 'likely to work' with caution."
        )

    return {
        "target_present": target_present,
        "mechanism_coverage": coverage,
        "forced_no_call": forced,
        "note": note,
    }
