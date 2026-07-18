"""Thin Gradio wrapper over Caveat's predictor.

This exists to satisfy the challenge's explicit "working Streamlit or Gradio demo"
requirement with minimal surface area. The polished product UI is the Next.js app;
this is the spec-compliant, zero-friction fallback that returns likely to fail /
likely to work / no-call per drug, with calibrated confidence, an evidence
category, and the mandatory lab-confirmation banner.

Run (from repo root):
    python api/gradio_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ml"))

from caveat_ml.config import CAVEAT_MESSAGE  # noqa: E402
from caveat_ml.predict import load_bundle, report_for_sample  # noqa: E402

ARTIFACTS_DIR = os.environ.get("CAVEAT_ARTIFACTS", str(REPO_ROOT / "ml" / "artifacts"))

_CALL_LABEL = {
    "likely_to_fail": "🔴 LIKELY TO FAIL",
    "likely_to_work": "🟢 LIKELY TO WORK",
    "no_call": "⚪ NO-CALL",
}


def _rows(genome_id: str):
    bundle = load_bundle(ARTIFACTS_DIR)
    report = report_for_sample(bundle, genome_id)
    rows = []
    for r in report.drug_results:
        genes = ", ".join(f"{f.gene}{'' if f.curated else ' (assoc.)'}" for f in r.supporting_features) or "—"
        note = r.no_call_reason.replace("_", " ") if r.no_call_reason else ""
        conf = "—" if r.confidence is None else f"{r.confidence:.2f}"
        rows.append([
            r.display_name,
            _CALL_LABEL.get(r.call, r.call),
            conf,
            f"type {r.evidence_type}",
            genes,
            note,
        ])
    return rows


def build_app():
    import gradio as gr

    bundle = load_bundle(ARTIFACTS_DIR)
    genome_ids = list(bundle["samples"].keys())

    with gr.Blocks(title="Caveat — Genome Firewall") as demo:
        gr.Markdown("# 🧬 Caveat — Genome Firewall\n"
                    "Defensive genome-to-antibiotic-response prediction for *Klebsiella pneumoniae*.")
        gr.Markdown(f"> ⚠️ **{CAVEAT_MESSAGE}**")
        with gr.Row():
            gid = gr.Dropdown(genome_ids, value=genome_ids[0] if genome_ids else None,
                              label="Held-out genome")
            go = gr.Button("Predict", variant="primary")
        table = gr.Dataframe(
            headers=["Antibiotic", "Call", "Confidence", "Evidence", "Supporting genes", "No-call reason"],
            interactive=False,
            wrap=True,
        )
        go.click(_rows, inputs=gid, outputs=table)
        if genome_ids:
            demo.load(_rows, inputs=gid, outputs=table)
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860)
