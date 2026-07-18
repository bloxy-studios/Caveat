"""Synthetic Klebsiella-like dataset generator.

Lets the whole train -> calibrate -> conformal -> predict pipeline run and be
tested WITHOUT a local AMRFinderPlus install or the (not-yet-released) fixed
challenge dataset. Swap this for the real BV-BRC loader when the data arrives —
the downstream code is identical.

Design choices that make it a faithful stand-in:
  * Clonal group structure: each group is an MDR-clone-like profile, so labels
    correlate within group and a grouped split genuinely matters.
  * Near-duplicates within groups exercise de-duplication.
  * Ciprofloxacin's dominant driver (chromosomal QRDR) is a HIDDEN variable not
    exposed as a feature -> the model stays honestly uncertain -> no-calls.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import ANTIBIOTIC_PANEL, FAMILY_CLASS, GENE_FAMILIES
from .features import EvidenceList, build_feature_matrix

CARBAPENEMASES = ["blaKPC", "blaNDM", "blaOXA-48-like", "blaVIM", "blaIMP"]
ESBL_STRONG = ["blaCTX-M", "blaCMY"]
GENT_GENES = ["aac(3)", "ant(2'')", "armA", "rmtB", "rmtC"]
CIPRO_PLASMID = ["qnrA", "qnrB", "qnrS", "aac(6')-Ib-cr", "qepA"]


def _amrfinder_like_frame(families: List[str], partial: bool = False) -> pd.DataFrame:
    rows = []
    for i, fam in enumerate(families):
        cov = 72.0 if (partial and i == 0) else 100.0
        ident = 88.0 if (partial and i == 0) else float(np.round(99.0 + np.random.rand(), 2))
        rows.append(
            {
                "gene_symbol": fam,  # round-trips through symbol_to_family
                "sequence_name": f"{fam} determinant",
                "scope": "core",
                "element_type": "AMR",
                "element_subtype": "AMR",
                "cls": FAMILY_CLASS.get(fam, ""),
                "subclass": "",
                "method": "EXACTX",
                "pct_coverage": cov,
                "pct_identity": ident,
                "contig": f"contig_{i}",
                "start": "1",
                "stop": "900",
                "closest_name": fam,
            }
        )
    if not rows:
        cols = ["gene_symbol", "sequence_name", "scope", "element_type", "element_subtype",
                "cls", "subclass", "method", "pct_coverage", "pct_identity", "contig",
                "start", "stop", "closest_name"]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def _group_profile(rng: np.random.Generator) -> Dict[str, object]:
    """A clone-level carriage profile."""
    carried = set()
    if rng.random() < 0.55:
        carried.add(rng.choice(CARBAPENEMASES))
    if rng.random() < 0.6:
        carried.add(rng.choice(ESBL_STRONG + ["blaSHV", "blaTEM"]))
    if rng.random() < 0.5:
        carried.add(rng.choice(GENT_GENES + ["aac(6')"]))
    if rng.random() < 0.5:
        carried.add("dfrA")
    if rng.random() < 0.5:
        carried.add(rng.choice(["sul1", "sul2"]))
    if rng.random() < 0.4:
        carried.add(rng.choice(CIPRO_PLASMID))
    if rng.random() < 0.3:
        carried.add("fosA")  # frequently intrinsic in Kp
    hidden_qrdr_prop = 0.85 if (carried & set(CARBAPENEMASES)) else 0.25
    return {"carried": carried, "hidden_qrdr_prop": hidden_qrdr_prop}


def _labels_for(fams: set, hidden_qrdr: bool, rng: np.random.Generator) -> Dict[str, int]:
    def flip(v: int, p: float = 0.04) -> int:
        return 1 - v if rng.random() < p else v

    carbapenemase = bool(fams & set(CARBAPENEMASES))
    esbl = bool(fams & set(ESBL_STRONG)) or ("blaSHV" in fams and rng.random() < 0.5) \
        or ("blaTEM" in fams and rng.random() < 0.3)
    gent = bool(fams & set(GENT_GENES))
    sxt = ("dfrA" in fams) or ({"sul1", "sul2"} & fams and rng.random() < 0.3)
    cipro = hidden_qrdr or (bool(fams & set(CIPRO_PLASMID)) and rng.random() < 0.6)

    return {
        "meropenem": flip(int(carbapenemase)),
        "ceftazidime": flip(int(esbl or carbapenemase)),
        "gentamicin": flip(int(gent)),
        "trimethoprim_sulfamethoxazole": flip(int(bool(sxt))),
        "ciprofloxacin": flip(int(cipro)),
    }


def generate(
    n_groups: int = 70,
    seed: int = 13,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, EvidenceList]]:
    """Return (feature_matrix, labels_df[R=1/S=0], groups, evidence_map)."""
    rng = np.random.default_rng(seed)
    per_genome: Dict[str, pd.DataFrame] = {}
    label_rows: Dict[str, Dict[str, int]] = {}
    group_of: Dict[str, int] = {}

    gid_counter = 0
    for g in range(1, n_groups + 1):
        prof = _group_profile(rng)
        carried = prof["carried"]
        n_members = int(rng.integers(3, 9))
        for _ in range(n_members):
            gid = f"KP{gid_counter:04d}"
            gid_counter += 1
            fams = set()
            # inherit carried genes with occasional dropout
            for fam in carried:
                if rng.random() < 0.85:
                    fams.add(fam)
            # occasional gain of a random family
            if rng.random() < 0.2:
                fams.add(rng.choice(GENE_FAMILIES))
            # near-duplicate members occasionally identical (exercise dedup)
            partial = rng.random() < 0.03
            per_genome[gid] = _amrfinder_like_frame(sorted(fams), partial=partial)
            hidden_qrdr = rng.random() < prof["hidden_qrdr_prop"]
            label_rows[gid] = _labels_for(fams, hidden_qrdr, rng)
            group_of[gid] = g

    feature_matrix, evidence_map = build_feature_matrix(per_genome)
    labels_df = pd.DataFrame.from_dict(label_rows, orient="index")[ANTIBIOTIC_PANEL]
    labels_df.index.name = "genome_id"
    groups = pd.Series(group_of, name="group").reindex(feature_matrix.index)
    return feature_matrix, labels_df, groups, evidence_map
