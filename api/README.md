# Caveat — API service (`api/`)

FastAPI service that serves the `/predict` contract to the Next.js UI and a thin
Gradio wrapper. The heavy ML lives in `../ml` (the `caveat_ml` package); this
service loads a trained artifact bundle and returns a typed `GenomeReport`.

## Run

```bash
# from repo root
pip install -r api/requirements.txt                 # REST API (lean)
python -m caveat_ml.train --data synth --out ml/artifacts   # if not already trained
uvicorn api.main:app --reload --port 8000           # REST API -> http://localhost:8000/docs

# optional Gradio demo (extra dep)
pip install -r api/requirements-demo.txt
python api/gradio_app.py                             # Gradio -> http://localhost:7860
```

Config via env:
- `CAVEAT_ARTIFACTS=/path/to/artifacts` — artifact bundle location.
- `CAVEAT_CORS_ORIGINS=https://your-app.vercel.app,https://...` — comma-separated allowed origins (defaults to `localhost:3000`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | model version, species, panel, alpha |
| GET | `/drugs` | drug knowledge base (class, molecular target, mechanisms) |
| GET | `/genomes` | held-out sample genomes for the demo dropdown |
| POST | `/predict` | `{genome_id}` or `{features}` → `GenomeReport` |

### `POST /predict`

```json
{ "genome_id": "KP0015", "alpha": 0.10 }
```

Returns per-drug results with `call` (`likely_to_fail`/`likely_to_work`/`no_call`),
calibrated `confidence`, `p_resistant`, conformal `prediction_set`, `evidence_type`
(`i`/`ii`/`iii`), `supporting_features`, `target_gate`, and the mandatory `caveat`.

The default path scores **precomputed held-out genomes** (fast + honest for the
demo). Live FASTA upload — running AMRFinderPlus server-side via
`caveat_ml.genome_reader.run_amrfinder` — is a documented extension.

> ⚠️ Research prototype. Decision support only; confirm every result with standard
> laboratory testing.
