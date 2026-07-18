"""Caveat ML backbone — defensive genome-to-antibiotic-response prediction.

A research prototype for Klebsiella pneumoniae (Hack-Nation Challenge 06,
"Genome Firewall"). Turns a quality-checked assembled FASTA into a per-drug
response report (likely to fail / likely to work / no-call) with calibrated
confidence and a transparent evidence ledger.

Strictly defensive: predicts and explains resistance that already exists.
Never designs, modifies, or optimizes an organism.
"""

__version__ = "0.1.0"
