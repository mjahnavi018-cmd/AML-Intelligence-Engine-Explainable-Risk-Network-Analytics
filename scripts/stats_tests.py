"""Statistical tests used for the pre-specified hypotheses (H1-H5).

Only these tests are run; all results are reported regardless of significance,
and Holm's step-down correction is applied across the family of tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mann_whitney(x: pd.Series, y: pd.Series, alternative: str = "two-sided", n_boot: int = 2000,
                 seed: int = 42) -> dict:
    """Mann-Whitney U with rank-biserial effect size and bootstrap CI of the median difference."""
    x, y = np.asarray(x.dropna(), float), np.asarray(y.dropna(), float)
    u, p = stats.mannwhitneyu(x, y, alternative=alternative)
    r_rb = 2 * u / (len(x) * len(y)) - 1          # P(X>Y) - P(X<Y)
    rng = np.random.default_rng(seed)
    diffs = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(y, len(y))) for _ in range(n_boot)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n_group1": len(x), "n_group2": len(y), "median_group1": float(np.median(x)),
            "median_group2": float(np.median(y)), "statistic": float(u), "p_value": float(p),
            "effect_size": float(r_rb), "effect_size_name": "rank-biserial r",
            "estimate": float(np.median(x) - np.median(y)), "ci_low": float(lo), "ci_high": float(hi),
            "estimate_name": "difference in medians (group1 - group2)"}


def two_proportions(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Fisher exact test with odds ratio and Woolf (log) 95% CI (Haldane correction if a cell is 0)."""
    table = np.array([[k1, n1 - k1], [k2, n2 - k2]], float)
    _, p = stats.fisher_exact(table.astype(int))
    t = table + 0.5 if (table == 0).any() else table
    or_ = (t[0, 0] * t[1, 1]) / (t[0, 1] * t[1, 0])
    se = np.sqrt((1 / t).sum())
    return {"n_group1": n1, "n_group2": n2, "rate_group1": k1 / n1, "rate_group2": k2 / n2,
            "statistic": float(or_), "p_value": float(p), "effect_size": float(or_), "effect_size_name": "odds ratio",
            "estimate": k1 / n1 - k2 / n2, "ci_low": float(np.exp(np.log(or_) - 1.96 * se)),
            "ci_high": float(np.exp(np.log(or_) + 1.96 * se)), "estimate_name": "difference in rates; CI is for OR"}


def mcnemar(a: pd.Series, b: pd.Series) -> dict:
    """Exact McNemar test on paired binary outcomes (e.g. detected by A vs by B)."""
    a, b = a.astype(bool), b.astype(bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {"statistic": float(n10 - n01), "p_value": float(p), "effect_size": n10 / max(n01, 1),
            "effect_size_name": "discordant ratio (A only / B only)", "estimate": n10 - n01,
            "estimate_name": f"detected only by A ({n10}) minus only by B ({n01})", "ci_low": np.nan, "ci_high": np.nan,
            "n_group1": int(a.sum()), "n_group2": int(b.sum())}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the expected (reference) sample."""
    e = expected.dropna().to_numpy(float)
    a = actual.dropna().to_numpy(float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(((pa - pe) * np.log(pa / pe)).sum())
