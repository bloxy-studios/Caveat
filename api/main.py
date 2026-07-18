"""Caveat FastAPI service — serves the /predict contract to the Next.js UI and
the Gradio wrapper.

Run (from repo root):
    uvicorn api.main:app --reload --port 8000

The heavy ML lives in ../ml (the `caveat_ml` package). This service loads a
trained artifact bundle and returns a typed GenomeReport. AMRFinderPlus is NOT
run per request in the default path — the demo scores precomputed held-out
genomes (fast + honest). Live FASTA upload is a documented extension.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Make the sibling ml/ package importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

from caveat_ml.config import CAVEAT_MESSAGE  # noqa: E402
from caveat_ml.predict import BundleNotFound, load_bundle, report_for_features, report_for_sample  # noqa: E402
from caveat_ml.schemas import GenomeReport, PredictRequest  # noqa: E402

ARTIFACTS_DIR = os.environ.get("CAVEAT_ARTIFACTS", str(ML_DIR / "artifacts"))

# CORS origins are configurable for deployment: set CAVEAT_CORS_ORIGINS to a
# comma-separated list (e.g. "https://caveat.vercel.app"). Defaults to local dev.
_DEFAULT_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
_CORS_ORIGINS = [o.strip() for o in os.environ.get("CAVEAT_CORS_ORIGINS", "").split(",") if o.strip()] \
    or _DEFAULT_ORIGINS

app = FastAPI(
    title="Caveat — Genome Firewall API",
    version="0.1.0",
    description="Defensive genome-to-antibiotic-response decision support for Klebsiella pneumoniae. "
    "Research prototype — every result must be confirmed with standard laboratory testing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bundle():
    try:
        return load_bundle(ARTIFACTS_DIR)
    except BundleNotFound as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health():
    try:
        b = load_bundle(ARTIFACTS_DIR)
        meta = b["meta"]
        return {
            "status": "ok",
            "model_version": meta.get("model_version"),
            "species": meta.get("species"),
            "panel": meta.get("panel"),
            "alpha": meta.get("alpha"),
            "n_sample_genomes": len(b["samples"]),
        }
    except BundleNotFound as exc:
        return {"status": "no_model", "detail": str(exc)}


@app.get("/drugs")
def drugs():
    return _bundle()["kb"].get("drugs", {})


@app.get("/genomes")
def genomes() -> List[dict]:
    b = _bundle()
    return [
        {"genome_id": gid, "true_labels": s.get("true_labels", {})}
        for gid, s in b["samples"].items()
    ]


@app.post("/predict", response_model=GenomeReport)
def predict(req: PredictRequest) -> GenomeReport:
    b = _bundle()
    if req.features:
        gid = req.genome_id or "user_supplied"
        return report_for_features(b, gid, req.features, None, req.alpha)
    if req.genome_id:
        try:
            return report_for_sample(b, req.genome_id, req.alpha)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail="Provide either 'genome_id' or 'features'.")


@app.get("/")
def root():
    return {"service": "caveat", "caveat": CAVEAT_MESSAGE, "docs": "/docs"}
