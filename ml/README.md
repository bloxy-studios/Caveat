# Caveat — ML backbone (`ml/`)

Defensive genome-to-antibiotic-response prediction for *Klebsiella pneumoniae*
(Hack-Nation Challenge 06, "Genome Firewall").

Turns a quality-checked assembled FASTA into a per-drug report — **likely to
fail / likely to work / no-call** — where every call carries a *calibrated*
confidence and a transparent evidence ledger. Strictly defensive: it predicts and
explains resistance that already exists and never designs or modifies an organism.

## Quickstart

```bash
cd ml
pip install -r requirements.txt

# Train on the built-in synthetic dataset (no AMRFinderPlus / real data needed)
python -m caveat_ml.train --data synth --out artifacts

# Score a held-out genome
python -m caveat_ml.predict --artifacts artifacts --list
python -m caveat_ml.predict --artifacts artifacts --genome KP0015
```

The synthetic generator (`caveat_ml/synth.py`) is a faithful stand-in for the
real BV-BRC data so the whole train → calibrate → conformal → predict flow runs
anywhere. Swap it for the real loader when the fixed dataset arrives — downstream
code is unchanged.

## The three modules

| Module | Files | What it does |
|---|---|---|
| **1. Genome Reader** | `genome_reader.py`, `features.py` | Runs AMRFinderPlus (`-n --organism Klebsiella --plus`), parses the TSV (tolerant to version drift), and builds a fixed, versioned feature vector (gene-family presence + class counts + annotation-quality flags). Emits `feature_spec.json`. |
| **2. Predictor** | `predictor.py`, `drug_kb.py`, `target_gate.py`, `grouping.py` | One L2-regularized logistic regression per antibiotic. A deterministic **molecular-target gate** never lets "no marker found" become "likely to work". Mash/Jaccard **de-duplication + grouped split** prevent memorization. |
| **3. Decision Report** | `calibration.py`, `conformal.py`, `predictor.py`, `schemas.py` | Isotonic calibration (+ Brier/reliability) and **Mondrian class-conditional conformal prediction** turn scores into `{R}`/`{S}`/`{R,S}`=no-call/`{}`=OOD-no-call. Every result is typed **i** (curated gene), **ii** (association), or **iii** (no signal) and carries the mandatory lab-confirmation caveat. |

## Why the design maps to the rubric

- **Calibrated confidence + no-call** — isotonic + conformal, fitted on separate
  group-disjoint slices so the coverage guarantee is valid.
- **Honest generalization** — metrics reported on a *genetically grouped* hold-out;
  a random-vs-grouped AUROC ablation is printed to expose inflation.
- **Honest explanations** — curated resistance genes (type i) are separated from
  statistical associations (type ii); the same gene can be type i for one drug and
  type ii for another (e.g. `blaCMY`: cause for ceftazidime, association for meropenem).
- **Coverage honesty** — ciprofloxacin's dominant driver (chromosomal gyrA/parC)
  is outside AMRFinderPlus gene-mode coverage for Klebsiella, so the target gate
  pushes it to no-call rather than guessing. This is stated, not hidden.

## Panel

`meropenem`, `ceftazidime`, `gentamicin`, `trimethoprim_sulfamethoxazole`,
`ciprofloxacin` (the deliberate coverage-honesty case).

## Artifacts (`artifacts/`)

`models.joblib` (per-drug models), `feature_spec.json`, `metrics.json`,
`meta.json`, `sample_genomes.json` (held-out demo genomes).

> ⚠️ Research prototype. Predictions from historical genome data do not prove the
> system is safe or suitable for clinical use. Every report must be confirmed with
> standard laboratory testing.
