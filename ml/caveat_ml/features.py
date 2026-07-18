"""Module 1 (part B): DNA annotations -> model features.

Produces a fixed, versioned feature vector per genome:
  presence__<family>   binary presence of a curated AMR gene family
  classcount__<class>  number of distinct families detected in a drug class
  annq__min_identity   annotation-quality: lowest %identity among curated hits
  annq__any_partial    1 if any curated hit is partial/low-coverage

It also emits an *evidence list* per genome (family, method, %id, %cov) that
powers the transparent ledger in the decision report.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .config import (
    CLASS_LIST,
    FAMILY_CLASS,
    GENE_FAMILIES,
    SCHEMA_VERSION,
    symbol_to_family,
)

EvidenceList = List[Dict[str, object]]


def get_feature_columns() -> List[str]:
    """The fixed, ordered feature-matrix column list."""
    cols = [f"presence__{fam}" for fam in GENE_FAMILIES]
    cols += [f"classcount__{cls}" for cls in CLASS_LIST]
    cols += ["annq__min_identity", "annq__any_partial"]
    return cols


def feature_row_from_amrfinder(df: pd.DataFrame) -> Tuple[Dict[str, float], EvidenceList]:
    """Turn one genome's parsed AMRFinder frame into (feature_dict, evidence_list)."""
    presence = {fam: 0 for fam in GENE_FAMILIES}
    class_families: Dict[str, set] = {cls: set() for cls in CLASS_LIST}
    evidence: EvidenceList = []
    identities: List[float] = []
    any_partial = 0

    for _, row in df.iterrows():
        # Only AMR determinants count as curated features; STRESS/VIRULENCE ignored.
        etype = str(row.get("element_type", "")).upper()
        if etype and etype not in ("AMR", ""):
            continue
        fam = symbol_to_family(str(row.get("gene_symbol", "")))
        if fam is None or fam not in presence:
            continue
        presence[fam] = 1
        cls = FAMILY_CLASS.get(fam)
        if cls in class_families:
            class_families[cls].add(fam)
        pid = row.get("pct_identity")
        pcov = row.get("pct_coverage")
        pid = float(pid) if pd.notna(pid) else None
        pcov = float(pcov) if pd.notna(pcov) else None
        if pid is not None:
            identities.append(pid)
        if (pid is not None and pid < 90.0) or (pcov is not None and pcov < 90.0):
            any_partial = 1
        evidence.append(
            {
                "gene": fam,
                "gene_symbol": str(row.get("gene_symbol", "")),
                "method": str(row.get("method", "")) or None,
                "pct_identity": pid,
                "pct_coverage": pcov,
                "cls": cls,
                "subclass": str(row.get("subclass", "")) or None,
            }
        )

    feat: Dict[str, float] = {}
    for fam in GENE_FAMILIES:
        feat[f"presence__{fam}"] = float(presence[fam])
    for cls in CLASS_LIST:
        feat[f"classcount__{cls}"] = float(len(class_families[cls]))
    feat["annq__min_identity"] = float(min(identities)) if identities else 100.0
    feat["annq__any_partial"] = float(any_partial)
    return feat, evidence


def build_feature_matrix(
    per_genome: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, EvidenceList]]:
    """Vectorize many genomes -> (feature DataFrame [fixed columns], evidence map)."""
    cols = get_feature_columns()
    rows: Dict[str, Dict[str, float]] = {}
    evidence_map: Dict[str, EvidenceList] = {}
    for gid, df in per_genome.items():
        feat, ev = feature_row_from_amrfinder(df)
        rows[gid] = feat
        evidence_map[gid] = ev
    matrix = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=cols).fillna(0.0)
    matrix.index.name = "genome_id"
    return matrix, evidence_map


def feature_spec(annotator_version: str = "unknown", db_version: str = "unknown") -> Dict[str, object]:
    """A machine-readable description of the feature contract (Module 1 requirement)."""
    columns = []
    for fam in GENE_FAMILIES:
        columns.append(
            {"name": f"presence__{fam}", "dtype": "binary", "source": "gene_family", "family": fam,
             "drug_class": FAMILY_CLASS.get(fam)}
        )
    for cls in CLASS_LIST:
        columns.append({"name": f"classcount__{cls}", "dtype": "int", "source": "class_count", "drug_class": cls})
    columns.append({"name": "annq__min_identity", "dtype": "float", "source": "annotation_quality"})
    columns.append({"name": "annq__any_partial", "dtype": "binary", "source": "annotation_quality"})
    return {
        "schema_version": SCHEMA_VERSION,
        "annotator": "AMRFinderPlus",
        "annotator_version": annotator_version,
        "database_version": db_version,
        "organism_flag": "Klebsiella",
        "n_columns": len(columns),
        "columns": columns,
        "notes": (
            "Missing determinant = 0, never NaN. Alleles collapse to family node. "
            "Klebsiella gene mode does not emit gyrA/parC point mutations -> chromosomal "
            "fluoroquinolone resistance is intentionally out of feature coverage."
        ),
    }


def row_to_vector(feature_row: Dict[str, float], columns: List[str]) -> List[float]:
    """Order a feature dict into the model's expected column vector."""
    return [float(feature_row.get(c, 0.0)) for c in columns]
