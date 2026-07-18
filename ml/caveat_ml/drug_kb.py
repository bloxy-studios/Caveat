"""Loader + helpers for the antimicrobial drug knowledge base (data/drug_kb.json).

The KB drives two things:
  1. Evidence typing (which detected families are *curated causes* for a drug).
  2. The deterministic molecular-target gate (mechanism coverage per drug).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set

_DEFAULT_KB = Path(__file__).resolve().parent.parent / "data" / "drug_kb.json"


@lru_cache(maxsize=4)
def load_drug_kb(path: str | None = None) -> Dict:
    kb_path = Path(path) if path else _DEFAULT_KB
    with open(kb_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def drug_entry(drug: str, kb: Dict | None = None) -> Dict:
    kb = kb or load_drug_kb()
    try:
        return kb["drugs"][drug]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Drug '{drug}' not found in knowledge base") from exc


def curated_families(drug: str, kb: Dict | None = None) -> Set[str]:
    """Families that are curated resistance causes for this drug (=> evidence type i)."""
    entry = drug_entry(drug, kb)
    return {m["family"] for m in entry.get("primary_mechanism_families", [])}


def association_families(drug: str, kb: Dict | None = None) -> Set[str]:
    """Families correlated with, but not a curated cause of, resistance (=> type ii)."""
    return set(drug_entry(drug, kb).get("association_families", []))


def mechanism_coverage(drug: str, kb: Dict | None = None) -> str:
    """'full' | 'partial' | 'none' — can our feature set interrogate the primary mechanism?"""
    return drug_entry(drug, kb).get("mechanism_coverage", "full")


def molecular_target_str(drug: str, kb: Dict | None = None) -> str:
    tgt = drug_entry(drug, kb).get("molecular_target", {})
    name = tgt.get("name", "unknown target")
    genes = tgt.get("genes", [])
    return f"{name} ({', '.join(genes)})" if genes else name


def mechanism_note(drug: str, family: str, kb: Dict | None = None) -> str:
    for m in drug_entry(drug, kb).get("primary_mechanism_families", []):
        if m["family"] == family:
            return m.get("mechanism", "")
    return ""


def all_drugs(kb: Dict | None = None) -> List[str]:
    kb = kb or load_drug_kb()
    return list(kb["drugs"].keys())
