"""Render a run directory into the results-first README table + accuracy/cost Pareto plot.

The report leads with the table (task success, contamination, staleness, tokens, $) and the
anchor block (calibration / oracle / placebo) so a reader sees the credibility gates before the
findings. Every metric prints as `mean [lo, hi]` — never a bare point estimate.

Stubbed: needs a completed run. The formatting helpers are the only new logic here.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def render(run_dir: str) -> Path:
    """Read data/runs/<run>/ and write report/out/{results.md, pareto.png}. TODO(milestone-5).

    Must include: the anchor validation block first (pass/fail + numbers), then the config
    table with CIs, then the accuracy-vs-cost Pareto scatter. If anchors failed, render a
    prominent VOID banner instead of findings."""
    raise NotImplementedError("TODO(milestone-5): format run metrics -> markdown table + Pareto plot")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a memprobe run report.")
    ap.add_argument("--run", default="data/runs/latest")
    args = ap.parse_args()
    print(f"report written to {render(args.run)}")


if __name__ == "__main__":
    main()
