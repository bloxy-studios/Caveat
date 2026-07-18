"""De-duplication + grouped splitting by genomic relatedness.

Two distinct operations (both required by the brief):
  1. De-duplication: collapse near-identical genomes so the same strain can't sit
     in train and test.
  2. Grouped split: assign genomes to genetically-related groups and split BY
     group, so the test set contains groups never seen in training.

Distance backend: Mash (MinHash) when the binary is available and FASTAs are
provided; otherwise a Jaccard distance over binary gene-presence features (a
dependency-free fallback so the pipeline runs anywhere, including on synthetic
data). The fallback is clearly labelled in outputs.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import GroupShuffleSplit


def mash_available() -> bool:
    return shutil.which("mash") is not None


def jaccard_distance_matrix(feature_matrix: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """Jaccard distance over presence__* columns. 0 = identical gene content."""
    pres_cols = [c for c in feature_matrix.columns if c.startswith("presence__")]
    ids = list(feature_matrix.index)
    X = (feature_matrix[pres_cols].values > 0).astype(int)
    n = X.shape[0]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = X[i], X[j]
            inter = np.sum(a & b)
            union = np.sum(a | b)
            d = 0.0 if union == 0 else 1.0 - inter / union
            dist[i, j] = dist[j, i] = d
    return ids, dist


def mash_distance_matrix(fasta_paths: Dict[str, str]) -> Tuple[List[str], np.ndarray]:
    """Pairwise Mash distances. `fasta_paths` maps genome_id -> FASTA path."""
    if not mash_available():
        raise RuntimeError("mash binary not found")
    ids = list(fasta_paths.keys())
    with tempfile.TemporaryDirectory() as tmp:
        sketch = Path(tmp) / "ref"
        subprocess.run(["mash", "sketch", "-o", str(sketch), *fasta_paths.values()], check=True)
        out = subprocess.run(
            ["mash", "dist", f"{sketch}.msh", f"{sketch}.msh"],
            check=True, capture_output=True, text=True,
        )
    path_to_id = {v: k for k, v in fasta_paths.items()}
    idx = {gid: i for i, gid in enumerate(ids)}
    n = len(ids)
    dist = np.zeros((n, n))
    for line in out.stdout.strip().splitlines():
        ref, qry, d, *_ = line.split("\t")
        gi, gj = path_to_id.get(ref), path_to_id.get(qry)
        if gi is None or gj is None:
            continue
        dist[idx[gi], idx[gj]] = float(d)
    return ids, dist


def compute_distance_matrix(
    feature_matrix: pd.DataFrame,
    fasta_paths: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], np.ndarray, str]:
    """Return (ids, distance_matrix, backend_name)."""
    if fasta_paths and mash_available():
        ids, dist = mash_distance_matrix(fasta_paths)
        return ids, dist, "mash"
    ids, dist = jaccard_distance_matrix(feature_matrix)
    return ids, dist, "jaccard_gene_presence(fallback)"


def deduplicate(ids: List[str], dist: np.ndarray, threshold: float) -> List[str]:
    """Greedy near-duplicate removal. Returns representative genome_ids."""
    kept: List[int] = []
    for i in range(len(ids)):
        if all(dist[i, j] > threshold for j in kept):
            kept.append(i)
    return [ids[i] for i in kept]


def assign_groups(ids: List[str], dist: np.ndarray, threshold: float) -> Dict[str, int]:
    """Single-linkage clustering at `threshold` -> genome_id -> group_id."""
    if len(ids) == 1:
        return {ids[0]: 1}
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="single")
    labels = fcluster(Z, t=threshold, criterion="distance")
    return {gid: int(lbl) for gid, lbl in zip(ids, labels)}


def grouped_split(
    groups: pd.Series,
    test_size: float = 0.25,
    calib_size: float = 0.25,
    seed: int = 7,
) -> Tuple[List[str], List[str], List[str]]:
    """Group-disjoint train / calibration / test split.

    calib_size is a fraction of the *non-test* remainder. All three sets are
    disjoint by group — essential for both honest generalization and valid
    conformal guarantees.
    """
    ids = list(groups.index)
    g = groups.values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(gss.split(ids, groups=g))
    rest_ids = [ids[i] for i in rest_idx]
    rest_groups = groups.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=calib_size, random_state=seed + 1)
    tr_idx, cal_idx = next(gss2.split(rest_ids, groups=rest_groups.values))
    train_ids = [rest_ids[i] for i in tr_idx]
    calib_ids = [rest_ids[i] for i in cal_idx]
    test_ids = [ids[i] for i in test_idx]
    return train_ids, calib_ids, test_ids
