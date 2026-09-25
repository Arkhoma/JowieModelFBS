"""Re-tune how hard the ratings lean on the preseason prior.

A better prior should earn a stronger grip. Reruns the production
walk-forward benchmark (tools/benchmark_vs_market.py) with different
margin-ridge `lambda_` / EPA `relative_lambda` values and scores each
against the closing line on the same games.

Usage: python tools/tune_prior_strength.py 5:4 8:4 12:4 8:6
       (each arg = margin_lambda:relative_lambda)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import benchmark_vs_market as bench  # noqa: E402
from cfbrank.epa_ridge import get_ep_model  # noqa: E402

EARLY_WEEK = 8


def run(margin_lambda: float, relative_lambda: float, market, totals):
    bench.MARGIN_PARAMS = {**bench.MARGIN_PARAMS, "lambda_": margin_lambda}
    bench.RELATIVE_LAMBDA = relative_lambda
    return pd.concat([bench.walk_season(s, market, totals)
                      for s in bench.SEASONS], ignore_index=True)


def main() -> None:
    market, totals = bench.load_home_lines(), bench.load_total_lines()
    get_ep_model()
    print(f"{'lambda':>7} {'rel':>4} {'n':>5} {'gap':>7} {'wk<=8':>7} "
          f"{'wk>8':>7} {'+/-':>6}", flush=True)
    for arg in sys.argv[1:]:
        lam, rel = (float(x) for x in arg.split(":"))
        frame = run(lam, rel, market, totals)
        gap = (frame.actual - frame.ours).abs() - (frame.actual - frame.line).abs()
        early = frame.week <= EARLY_WEEK
        ci = 1.96 * gap.std(ddof=1) / np.sqrt(len(gap))
        print(f"{lam:>7.1f} {rel:>4.1f} {len(frame):>5} {gap.mean():>+7.3f} "
              f"{gap[early].mean():>+7.3f} {gap[~early].mean():>+7.3f} "
              f"{ci:>6.3f}", flush=True)
        frame.to_csv(ROOT / "data" / f"tune_{lam:g}_{rel:g}.csv", index=False)


if __name__ == "__main__":
    main()
