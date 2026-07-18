"""Central configuration: the antibiotic panel, the AMR gene-family vocabulary,
family/class maps, and global constants used across the pipeline.

Keeping this in one place makes the FASTA->features path *documented and
repeatable* (a Module 1 requirement) and gives every artifact a single
versioned source of truth.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

SCHEMA_VERSION = "0.1.0"
SPECIES = "Klebsiella pneumoniae"
ORGANISM_FLAG = "Klebsiella"  # AMRFinderPlus --organism value

# Conformal significance level (alpha). 0.10 => target >=90% class-conditional coverage.
DEFAULT_ALPHA = 0.10

# The honest, small panel. Gene-presence-clean drugs + ciprofloxacin as the
# deliberate "coverage-honesty" case (chromosomal QRDR not called by AMRFinder
# in Klebsiella gene mode -> leans to no-call / statistical-only).
ANTIBIOTIC_PANEL: List[str] = [
    "meropenem",
    "ceftazidime",
    "gentamicin",
    "trimethoprim_sulfamethoxazole",
    "ciprofloxacin",
]

# Canonical AMR gene families we featurize (alleles collapse to family node).
# Order is fixed -> it defines the feature-matrix column order.
GENE_FAMILIES: List[str] = [
    # carbapenemases
    "blaKPC", "blaNDM", "blaOXA-48-like", "blaVIM", "blaIMP",
    # other beta-lactamases / ESBL / AmpC
    "blaCTX-M", "blaSHV", "blaTEM", "blaCMY", "blaOXA-1",
    # aminoglycoside-modifying enzymes + 16S methyltransferases
    "aac(3)", "aac(6')", "aph(3')", "aph(6)", "ant(2'')", "aadA",
    "armA", "rmtB", "rmtC",
    # fluoroquinolone (plasmid-mediated only; chromosomal QRDR is NOT here)
    "qnrA", "qnrB", "qnrS", "oqxAB", "qepA", "aac(6')-Ib-cr",
    # folate pathway
    "sul1", "sul2", "sul3", "dfrA",
    # misc / intrinsic / association features
    "fosA", "mcr", "catA", "tetA",
]

# family -> broad AMR class (drives class-count features and the synth generator)
FAMILY_CLASS: Dict[str, str] = {
    "blaKPC": "CARBAPENEM", "blaNDM": "CARBAPENEM", "blaOXA-48-like": "CARBAPENEM",
    "blaVIM": "CARBAPENEM", "blaIMP": "CARBAPENEM",
    "blaCTX-M": "CEPHALOSPORIN", "blaSHV": "BETA-LACTAM", "blaTEM": "BETA-LACTAM",
    "blaCMY": "CEPHALOSPORIN", "blaOXA-1": "BETA-LACTAM",
    "aac(3)": "AMINOGLYCOSIDE", "aac(6')": "AMINOGLYCOSIDE", "aph(3')": "AMINOGLYCOSIDE",
    "aph(6)": "AMINOGLYCOSIDE", "ant(2'')": "AMINOGLYCOSIDE", "aadA": "AMINOGLYCOSIDE",
    "armA": "AMINOGLYCOSIDE", "rmtB": "AMINOGLYCOSIDE", "rmtC": "AMINOGLYCOSIDE",
    "qnrA": "QUINOLONE", "qnrB": "QUINOLONE", "qnrS": "QUINOLONE", "oqxAB": "QUINOLONE",
    "qepA": "QUINOLONE", "aac(6')-Ib-cr": "QUINOLONE",
    "sul1": "SULFONAMIDE", "sul2": "SULFONAMIDE", "sul3": "SULFONAMIDE",
    "dfrA": "TRIMETHOPRIM",
    "fosA": "FOSFOMYCIN", "mcr": "COLISTIN", "catA": "PHENICOL", "tetA": "TETRACYCLINE",
}

# Classes used for class-count features (stable order).
CLASS_LIST: List[str] = [
    "CARBAPENEM", "BETA-LACTAM", "CEPHALOSPORIN", "AMINOGLYCOSIDE",
    "QUINOLONE", "SULFONAMIDE", "TRIMETHOPRIM", "FOSFOMYCIN",
    "COLISTIN", "PHENICOL", "TETRACYCLINE",
]

# Ordered regex rules mapping an AMRFinderPlus gene symbol to a canonical family.
# First match wins. Symbols that match nothing become association ("plus") noise
# and are ignored for the curated feature vector.
_OXA_48_LIKE = {"blaOXA-48", "blaOXA-181", "blaOXA-232", "blaOXA-204", "blaOXA-162", "blaOXA-244"}

_FAMILY_RULES: List[Tuple[str, str]] = [
    (r"^blaKPC", "blaKPC"),
    (r"^blaNDM", "blaNDM"),
    (r"^blaVIM", "blaVIM"),
    (r"^blaIMP", "blaIMP"),
    (r"^blaCTX-M", "blaCTX-M"),
    (r"^blaSHV", "blaSHV"),
    (r"^blaTEM", "blaTEM"),
    (r"^blaCMY", "blaCMY"),
    (r"^blaOXA-1$", "blaOXA-1"),
    (r"^aac\(6'\)-Ib-cr", "aac(6')-Ib-cr"),  # must precede generic aac(6')
    (r"^aac\(3\)", "aac(3)"),
    (r"^aac\(6'\)", "aac(6')"),
    (r"^aph\(3'\)", "aph(3')"),
    (r"^aph\(6\)", "aph(6)"),
    (r"^ant\(2''\)", "ant(2'')"),
    (r"^aadA", "aadA"),
    (r"^armA", "armA"),
    (r"^rmtB", "rmtB"),
    (r"^rmtC", "rmtC"),
    (r"^qnrA", "qnrA"),
    (r"^qnrB", "qnrB"),
    (r"^qnrS", "qnrS"),
    (r"^oqx[AB]", "oqxAB"),
    (r"^qepA", "qepA"),
    (r"^sul1", "sul1"),
    (r"^sul2", "sul2"),
    (r"^sul3", "sul3"),
    (r"^dfrA", "dfrA"),
    (r"^fosA", "fosA"),
    (r"^mcr", "mcr"),
    (r"^catA", "catA"),
    (r"^tet\(?A", "tetA"),
]

_COMPILED_RULES = [(re.compile(rx), fam) for rx, fam in _FAMILY_RULES]


def symbol_to_family(gene_symbol: str) -> str | None:
    """Collapse an AMRFinderPlus gene symbol (e.g. 'blaKPC-3') to a canonical
    family (e.g. 'blaKPC'). Returns None if the symbol isn't in our vocabulary."""
    if not gene_symbol:
        return None
    sym = gene_symbol.strip()
    if sym in _OXA_48_LIKE or sym.startswith("blaOXA-48"):
        return "blaOXA-48-like"
    for rx, fam in _COMPILED_RULES:
        if rx.search(sym):
            return fam
    return None


# The mandatory, non-negotiable message attached to every result.
CAVEAT_MESSAGE = (
    "Research prototype. This antibiotic-response report is decision support only "
    "and must be confirmed with standard laboratory testing before any treatment decision."
)
