"""Train Caveat's per-drug models and save a self-contained artifact bundle.

    python -m caveat_ml.train --data synth --out artifacts --alpha 0.10

Pipeline: load -> de-duplicate -> grouped split (train/calib/test) -> split calib
into isotonic/conformal slices -> per-drug fit -> evaluate on the grouped hold-out
with the full metric suite -> random-vs-grouped ablation -> save artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from . import __version__
from .calibration import brier, reliability
from .conformal import coverage_report
from .config import ANTIBIOTIC_PANEL, DEFAULT_ALPHA, SCHEMA_VERSION, SPECIES
from .drug_kb import load_drug_kb
from .features import feature_spec, get_feature_columns
from .grouping import assign_groups, compute_distance_matrix, deduplicate, grouped_split
from .predictor import fit_drug_model

DEDUP_THRESHOLD = 0.02      # near-identical gene content -> one representative
GROUP_THRESHOLD = 0.30      # single-linkage relatedness groups (fallback backend)


def _safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except Exception:
        return None


def _xy(feature_matrix, labels_df, ids: List[str], drug: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    sub = labels_df.loc[ids, drug].dropna()
    kept = list(sub.index)
    X = feature_matrix.loc[kept].values.astype(float)
    y = sub.values.astype(int)
    return X, y, kept


def _load_data(source: str):
    if source == "synth":
        from .synth import generate
        return generate()
    # real data hook: expects features.csv, labels.csv, groups.csv in `source` dir
    d = Path(source)
    fm = pd.read_csv(d / "features.csv", index_col=0)
    lb = pd.read_csv(d / "labels.csv", index_col=0)
    gp = pd.read_csv(d / "groups.csv", index_col=0).iloc[:, 0]
    return fm, lb, gp, {gid: [] for gid in fm.index}


def _random_auroc(feature_matrix, labels_df, drug, n_test, n_train, seed) -> Optional[float]:
    """AUROC under a naive RANDOM split, with train/test sizes matched to the
    grouped split. Paired with the real grouped-test AUROC, this isolates the
    inflation that comes purely from ignoring genetic relatedness when splitting."""
    X, y, ids = _xy(feature_matrix, labels_df, list(feature_matrix.index), drug)
    if len(np.unique(y)) < 2 or n_test < 1 or n_train < 1:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:n_test + n_train]
    if len(train_idx) < 2 or len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        return None
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[train_idx], y[train_idx])
    return _safe(roc_auc_score, y[test_idx], m.predict_proba(X[test_idx])[:, 1])


def train(data: str = "synth", out: str = "artifacts", alpha: float = DEFAULT_ALPHA, seed: int = 7) -> Dict:
    feature_matrix, labels_df, groups, evidence_map = _load_data(data)
    feature_columns = get_feature_columns()
    feature_matrix = feature_matrix.reindex(columns=feature_columns).fillna(0.0)
    kb = load_drug_kb()

    # 1) de-duplicate on genomic relatedness
    ids_all, dist, backend = compute_distance_matrix(feature_matrix)
    reps = deduplicate(ids_all, dist, DEDUP_THRESHOLD)
    fm = feature_matrix.loc[reps]
    lb = labels_df.loc[reps]
    grp = groups.loc[reps]

    # 2) relatedness groups + grouped split.
    # Use known clonal groups when available (synth / provided). For real data
    # with no group labels, derive them by clustering the distance matrix (the
    # same Mash-distance clustering used in production).
    rep_groups = grp.reindex(reps)
    if rep_groups.isna().any():
        idx = [ids_all.index(r) for r in reps]
        sub = dist[np.ix_(idx, idx)]
        rep_groups = pd.Series(assign_groups(reps, sub, GROUP_THRESHOLD)).reindex(reps)
    train_ids, calib_ids, test_ids = grouped_split(rep_groups, test_size=0.25, calib_size=0.30, seed=seed)

    # 3) split calibration into isotonic / conformal slices (group-disjoint)
    cal_groups = rep_groups.loc[calib_ids]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 5)
    iso_idx, conf_idx = next(gss.split(calib_ids, groups=cal_groups.values))
    iso_ids = [calib_ids[i] for i in iso_idx]
    conf_ids = [calib_ids[i] for i in conf_idx]

    models: Dict = {}
    metrics: Dict[str, Dict] = {}
    for drug in ANTIBIOTIC_PANEL:
        Xtr, ytr, _ = _xy(fm, lb, train_ids, drug)
        Xiso, yiso, _ = _xy(fm, lb, iso_ids, drug)
        Xcf, ycf, _ = _xy(fm, lb, conf_ids, drug)
        Xte, yte, _ = _xy(fm, lb, test_ids, drug)
        if len(np.unique(ytr)) < 2 or len(np.unique(yiso)) < 2 or len(np.unique(ycf)) < 2:
            metrics[drug] = {"status": "skipped_single_class_in_train_or_calib"}
            continue

        model = fit_drug_model(drug, Xtr, ytr, Xiso, yiso, Xcf, ycf, feature_columns, kb)
        models[drug] = model

        p_test = model.predict_calibrated(Xte)
        yhat = (p_test >= 0.5).astype(int)
        cov = coverage_report(model.conformal, p_test, yte, alpha)
        grouped_auroc = _safe(roc_auc_score, yte, p_test)  # the ONE grouped number
        metrics[drug] = {
            "status": "ok",
            "n_train": int(len(ytr)), "n_calib_iso": int(len(yiso)),
            "n_calib_conf": int(len(ycf)), "n_test": int(len(yte)),
            "prevalence_R_test": float(np.mean(yte)),
            "balanced_accuracy": _safe(balanced_accuracy_score, yte, yhat),
            "recall_resistant": _safe(recall_score, yte, yhat, pos_label=1, zero_division=0),
            "recall_susceptible": _safe(recall_score, yte, yhat, pos_label=0, zero_division=0),
            "f1_resistant": _safe(f1_score, yte, yhat, pos_label=1, zero_division=0),
            "auroc": grouped_auroc,
            "pr_auc": _safe(average_precision_score, yte, p_test),
            "brier": brier(yte, p_test),
            "reliability": reliability(yte, p_test),
            "conformal": cov,
            # ablation reuses the real grouped-test AUROC so the demo shows one
            # consistent number, contrasted with a size-matched random split.
            "ablation": {
                "grouped_auroc": grouped_auroc,
                "random_auroc": _random_auroc(fm, lb, drug, len(yte), len(ytr), seed),
            },
        }

    # persist artifacts
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, out_dir / "models.joblib")
    (out_dir / "feature_spec.json").write_text(json.dumps(feature_spec(), indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # sample held-out genomes for the demo (features + evidence + truth)
    sample = {}
    for gid in test_ids[:14]:
        sample[gid] = {
            "features": {k: float(v) for k, v in fm.loc[gid].to_dict().items()},
            "evidence": evidence_map.get(gid, []),
            "true_labels": {d: ("R" if int(v) == 1 else "S") for d, v in lb.loc[gid].items() if pd.notna(v)},
        }
    (out_dir / "sample_genomes.json").write_text(json.dumps(sample, indent=2))

    meta = {
        "schema_version": SCHEMA_VERSION,
        "model_version": __version__,
        "species": SPECIES,
        "panel": ANTIBIOTIC_PANEL,
        "alpha": alpha,
        "data_source": data,
        "distance_backend": backend,
        "dedup_threshold": DEDUP_THRESHOLD,
        "group_threshold": GROUP_THRESHOLD,
        "n_genomes_raw": int(len(ids_all)),
        "n_representatives": int(len(reps)),
        "n_groups": int(rep_groups.nunique()),
        "split": {"train": len(train_ids), "calib_iso": len(iso_ids),
                  "calib_conf": len(conf_ids), "test": len(test_ids)},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"meta": meta, "metrics": metrics}


def _print_summary(result: Dict) -> None:
    meta, metrics = result["meta"], result["metrics"]
    print(f"\nCaveat trained on {meta['data_source']} | backend={meta['distance_backend']}")
    print(f"genomes={meta['n_genomes_raw']} reps={meta['n_representatives']} "
          f"groups={meta['n_groups']} split={meta['split']}\n")
    hdr = f"{'drug':<32}{'balAcc':>7}{'recR':>6}{'recS':>6}{'AUROC':>7}{'PR':>6}{'Brier':>7}{'noCall':>8}{'covR':>6}{'covS':>6}"
    print(hdr); print("-" * len(hdr))
    for drug in meta["panel"]:
        m = metrics.get(drug, {})
        if m.get("status") != "ok":
            print(f"{drug:<32}{m.get('status',''):>50}")
            continue
        c = m["conformal"]

        def f(x, nd=2):
            return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "  -"
        print(f"{drug:<32}{f(m['balanced_accuracy']):>7}{f(m['recall_resistant']):>6}"
              f"{f(m['recall_susceptible']):>6}{f(m['auroc']):>7}{f(m['pr_auc']):>6}"
              f"{f(m['brier']):>7}{f(c['no_call_rate']):>8}{f(c['coverage_R']):>6}{f(c['coverage_S']):>6}")
    print("\nRandom-vs-grouped AUROC (honesty ablation):")
    for drug in meta["panel"]:
        ab = metrics.get(drug, {}).get("ablation", {})
        if ab.get("grouped_auroc") is not None and ab.get("random_auroc") is not None:
            print(f"  {drug:<32} grouped={ab['grouped_auroc']:.3f}  random={ab['random_auroc']:.3f}  "
                  f"(inflation={ab['random_auroc']-ab['grouped_auroc']:+.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Caveat per-drug models")
    ap.add_argument("--data", default="synth", help="'synth' or a dir with features.csv/labels.csv/groups.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    result = train(args.data, args.out, args.alpha, args.seed)
    _print_summary(result)
    print(f"\nArtifacts written to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
