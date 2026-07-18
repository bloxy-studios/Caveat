"""Module 1 (part A): The Genome Reader.

Runs AMRFinderPlus on an assembled FASTA and parses its TSV into a tidy frame.
Header names have drifted across AMRFinderPlus versions, so parsing is tolerant:
we map many known column spellings onto a small canonical schema.

The actual feature vectorization lives in features.py; this module only turns
raw sequence -> annotated determinants.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import ORGANISM_FLAG

# canonical field -> list of candidate raw header spellings (any version)
_HEADER_CANDIDATES: Dict[str, List[str]] = {
    "gene_symbol": ["Gene symbol", "Element symbol"],
    "sequence_name": ["Sequence name", "Element name"],
    "scope": ["Scope"],
    "element_type": ["Element type", "Type"],
    "element_subtype": ["Element subtype", "Subtype"],
    "cls": ["Class"],
    "subclass": ["Subclass"],
    "method": ["Method"],
    "pct_coverage": [
        "% Coverage of reference sequence", "% Coverage of reference",
        "Coverage of reference sequence",
    ],
    "pct_identity": [
        "% Identity to reference sequence", "% Identity to reference",
        "Identity to reference sequence",
    ],
    "contig": ["Contig id", "Contig"],
    "start": ["Start"],
    "stop": ["Stop"],
    "closest_name": ["Name of closest sequence", "Closest reference name"],
}


class AMRFinderNotInstalled(RuntimeError):
    pass


def amrfinder_available() -> bool:
    return shutil.which("amrfinder") is not None


def run_amrfinder(
    fasta_path: str,
    genome_id: str,
    out_dir: str = "artifacts/amrfinder",
    organism: str = ORGANISM_FLAG,
    threads: int = 4,
    plus: bool = True,
) -> str:
    """Run AMRFinderPlus in nucleotide mode. Returns the output TSV path.

    Command mirrors the documented default annotation path:
        amrfinder -n <fasta> --organism Klebsiella --plus --name <id> -o <tsv>
    """
    if not amrfinder_available():
        raise AMRFinderNotInstalled(
            "The 'amrfinder' binary was not found. Install AMRFinderPlus and its "
            "database (see https://github.com/ncbi/amr), or use precomputed features."
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tsv = out / f"{genome_id}.amrfinder.tsv"
    cmd = [
        "amrfinder", "-n", fasta_path,
        "--organism", organism,
        "--name", genome_id,
        "--threads", str(threads),
        "-o", str(tsv),
    ]
    if plus:
        cmd.append("--plus")
    subprocess.run(cmd, check=True)
    return str(tsv)


def parse_amrfinder_tsv(tsv_path: str) -> pd.DataFrame:
    """Parse an AMRFinderPlus TSV into the canonical schema (tolerant to version)."""
    raw = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    return normalize_amrfinder_frame(raw)


def normalize_amrfinder_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Map an already-loaded AMRFinderPlus frame onto the canonical columns."""
    colmap: Dict[str, str] = {}
    for canonical, candidates in _HEADER_CANDIDATES.items():
        for cand in candidates:
            if cand in raw.columns:
                colmap[cand] = canonical
                break
    out = raw.rename(columns=colmap)
    for canonical in _HEADER_CANDIDATES:
        if canonical not in out.columns:
            out[canonical] = ""
    keep = list(_HEADER_CANDIDATES.keys())
    out = out[keep].copy()
    for numeric in ("pct_coverage", "pct_identity"):
        out[numeric] = pd.to_numeric(out[numeric], errors="coerce")
    return out
