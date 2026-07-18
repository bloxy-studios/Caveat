"""Load a trained artifact bundle and produce antibiotic-response reports.

    python -m caveat_ml.predict --artifacts artifacts --genome KP0001
    python -m caveat_ml.predict --artifacts artifacts --list
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import joblib

from .config import DEFAULT_ALPHA
from .drug_kb import load_drug_kb
from .predictor import build_report
from .schemas import GenomeReport


class BundleNotFound(FileNotFoundError):
    pass


@lru_cache(maxsize=2)
def load_bundle(artifacts_dir: str = "artifacts") -> Dict:
    d = Path(artifacts_dir)
    models_path = d / "models.joblib"
    if not models_path.exists():
        raise BundleNotFound(f"No models.joblib in {d.resolve()} — run `python -m caveat_ml.train` first.")
    models = joblib.load(models_path)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    samples = json.loads((d / "sample_genomes.json").read_text()) if (d / "sample_genomes.json").exists() else {}
    return {"models": models, "meta": meta, "samples": samples, "kb": load_drug_kb()}


def report_for_features(
    bundle: Dict,
    genome_id: str,
    feature_row: Dict[str, float],
    evidence_list=None,
    alpha: Optional[float] = None,
) -> GenomeReport:
    alpha = alpha if alpha is not None else bundle.get("meta", {}).get("alpha", DEFAULT_ALPHA)
    return build_report(genome_id, feature_row, evidence_list, bundle["models"], alpha, bundle["kb"])


def report_for_sample(bundle: Dict, genome_id: str, alpha: Optional[float] = None) -> GenomeReport:
    sample = bundle["samples"].get(genome_id)
    if sample is None:
        raise KeyError(f"Genome '{genome_id}' not in sample store. Available: {list(bundle['samples'])[:10]}")
    return report_for_features(bundle, genome_id, sample["features"], sample.get("evidence"), alpha)


def main() -> None:
    ap = argparse.ArgumentParser(description="Caveat prediction")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--genome", help="sample genome_id to score")
    ap.add_argument("--list", action="store_true", help="list available sample genomes")
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()
    bundle = load_bundle(args.artifacts)
    if args.list or not args.genome:
        print("Sample genomes:", ", ".join(list(bundle["samples"].keys())))
        return
    report = report_for_sample(bundle, args.genome, args.alpha)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
