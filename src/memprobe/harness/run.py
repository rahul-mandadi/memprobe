"""Config-matrix runner — the top-level entrypoint (`memprobe` / `make bench`).

Orchestrates: generate scenarios -> for each config cell, run the agent over every user/seed
-> score deterministically -> aggregate with CIs -> VALIDATE ANCHORS (abort if void) -> emit a
run directory the report renderer consumes.

Stubbed at the orchestration level (needs the agent graph). The pieces it wires — generator,
policies.resolve, metrics, stats, anchors — are implemented and independently tested, so this
is assembly, not new logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memprobe.config import RunConfig


def run(config_path: str) -> Path:
    """Execute the full matrix and return the run directory. TODO(milestone-1..5).

    Order of operations (each step already has a home):
      1. cfg = RunConfig.load(config_path)
      2. for seed in cfg.scenarios.seeds: suite = generator.generate_suite(seed, params)
      3. wire cross-user contamination probes (each user's values -> others' must_not_contain)
      4. for mem_cfg in cfg.configs: policy = policies.resolve(mem_cfg);
         agent = graph.build_agent(policy, store, model); run every session; collect responses
      5. score with eval.metrics; aggregate per (config, seed) then eval.stats.mean_ci
      6. eval.anchors.validate(...) — if not ok, ABORT and print the failure (run is void)
      7. persist per-config metrics + usage to data/runs/<ts>/ and update data/runs/latest
    """
    _ = RunConfig.load(config_path)  # config parsing works today
    raise NotImplementedError(
        "TODO: wire generator -> agent.graph -> metrics -> stats -> anchors (milestones 1-5)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the memprobe ablation matrix.")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    out = run(args.config)
    print(f"run written to {out}")


if __name__ == "__main__":
    main()
