"""Mondrian (class-conditional) inductive conformal prediction for binary R/S.

This is Caveat's no-call engine. For each drug we compute nonconformity scores on
a held-out calibration slice, bucketed by TRUE class (Mondrian), then for a test
genome we build a prediction SET over {R, S}:

    {R}    -> likely to fail
    {S}    -> likely to work
    {R,S}  -> no-call (conflicting evidence)
    {}     -> no-call (unlike training data / out-of-distribution)

Bucketing by true class gives a per-class coverage guarantee (~1 - alpha), which
is exactly what the rubric asks for: recall reported separately for resistant and
susceptible cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

RESISTANT_LABEL = 1  # R
SUSCEPTIBLE_LABEL = 0  # S


@dataclass
class MondrianBinaryConformal:
    """Nonconformity = 1 - p(true class). Scores stored per true class."""

    scores_R: np.ndarray = field(default_factory=lambda: np.array([]))
    scores_S: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, cal_p_resistant: np.ndarray, cal_labels: np.ndarray) -> "MondrianBinaryConformal":
        p = np.asarray(cal_p_resistant, dtype=float)
        y = np.asarray(cal_labels, dtype=int)
        # For a resistant (R) calibration point: p(true=R) = p  -> score = 1 - p
        self.scores_R = np.sort(1.0 - p[y == RESISTANT_LABEL])
        # For a susceptible (S) calibration point: p(true=S) = 1 - p -> score = p
        self.scores_S = np.sort(p[y == SUSCEPTIBLE_LABEL])
        return self

    @staticmethod
    def _p_value(score: float, cal_scores: np.ndarray) -> float:
        if cal_scores.size == 0:
            return 1.0  # no calibration data for this class -> be permissive
        n = cal_scores.size
        return (1.0 + int(np.sum(cal_scores >= score))) / (n + 1.0)

    def predict_set(self, p_resistant: float, alpha: float) -> List[str]:
        """Return the conformal prediction set (subset of ['R','S'])."""
        pset: List[str] = []
        # candidate R: test nonconformity = 1 - p ; compare against R-class scores
        if self._p_value(1.0 - p_resistant, self.scores_R) > alpha:
            pset.append("R")
        # candidate S: test nonconformity = p ; compare against S-class scores
        if self._p_value(p_resistant, self.scores_S) > alpha:
            pset.append("S")
        return pset

    def to_dict(self) -> Dict[str, List[float]]:
        return {"scores_R": self.scores_R.tolist(), "scores_S": self.scores_S.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, List[float]]) -> "MondrianBinaryConformal":
        obj = cls()
        obj.scores_R = np.array(d.get("scores_R", []), dtype=float)
        obj.scores_S = np.array(d.get("scores_S", []), dtype=float)
        return obj


def set_to_call(pset: List[str]) -> Tuple[str, str | None]:
    """Map a prediction set to (call, no_call_reason)."""
    s = set(pset)
    if s == {"R"}:
        return "likely_to_fail", None
    if s == {"S"}:
        return "likely_to_work", None
    if s == {"R", "S"}:
        return "no_call", "conflicting_evidence"
    return "no_call", "unlike_training_data"  # empty set


def coverage_report(
    conf: MondrianBinaryConformal,
    p_resistant: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    """Empirical per-class coverage, set sizes, no-call rate, singleton accuracy."""
    p = np.asarray(p_resistant, dtype=float)
    y = np.asarray(labels, dtype=int)
    covered_R = covered_S = n_R = n_S = 0
    set_sizes: List[int] = []
    no_calls = 0
    singleton_correct = singleton_total = 0
    for pi, yi in zip(p, y):
        pset = conf.predict_set(pi, alpha)
        set_sizes.append(len(pset))
        true_symbol = "R" if yi == RESISTANT_LABEL else "S"
        if true_symbol in pset:
            if yi == RESISTANT_LABEL:
                covered_R += 1
            else:
                covered_S += 1
        if yi == RESISTANT_LABEL:
            n_R += 1
        else:
            n_S += 1
        if len(pset) != 1:
            no_calls += 1
        else:
            singleton_total += 1
            if pset[0] == true_symbol:
                singleton_correct += 1
    n = len(y)
    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "coverage_R": (covered_R / n_R) if n_R else float("nan"),
        "coverage_S": (covered_S / n_S) if n_S else float("nan"),
        "avg_set_size": float(np.mean(set_sizes)) if set_sizes else float("nan"),
        "no_call_rate": (no_calls / n) if n else float("nan"),
        "accuracy_on_confident": (singleton_correct / singleton_total) if singleton_total else float("nan"),
        "n": n,
    }
