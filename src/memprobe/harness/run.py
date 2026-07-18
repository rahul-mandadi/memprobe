"""Config-matrix runner — the top-level entrypoint (`memprobe`).

Orchestrates: generate scenarios -> wire cross-user contamination probes -> for each config
cell, run the agent over every user/seed in session-lockstep -> score deterministically ->
aggregate with CIs -> VALIDATE ANCHORS (persist + report VOID and raise if they fail) ->
persist a run directory -> render the report.

Assembly, not new logic: every piece it wires (generator, policies.resolve, graph, metrics,
stats, anchors, render) is implemented and independently tested.

Design decisions here (ADR-0012):
- LOGICAL CLOCK: sessions are SESSION_SPACING_DAYS apart (day units), so the forgetting
  half-life (configured in days) has real bite inside an 8-session scenario. The clock lives
  in the harness because time is scenario semantics, not agent policy.
- SESSION-LOCKSTEP ORDER: all users run session s before any user runs s+1. The placebo
  anchor requires it — user A's probe-time read of neighbor B's namespace must see B's
  prior-session writes, which per-user-sequential order would not guarantee.
- SCRAMBLE MAPPING: user i reads user (i+1) mod n — a fixed-point-free permutation, so under
  the placebo no user is ever served their own memory.
- CONTAMINATION WIRING: every OTHER user's current value for a probed key (minus the user's
  own expected value) lands in must_not_contain. With small closed vocabularies this reads
  as "asserted a value with no support in this user's own history" — an honest
  unsupported-assertion channel; the statistical cross-user leak gate is the placebo anchor
  (ADR-0013). A per-user unique-token tracer is the sharper v2 instrument.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

from memprobe.agent.graph import build_agent
from memprobe.agent.models import build_model
from memprobe.agent.policies import resolve
from memprobe.config import Anchor, MemConfig, RunConfig
from memprobe.eval import anchors as anchors_mod
from memprobe.eval.metrics import aggregate, chance_rate, score_probe
from memprobe.eval.stats import mean_ci
from memprobe.memory.retrieval import build_embedder
from memprobe.memory.store import LangGraphStore
from memprobe.scenarios.generator import generate_suite
from memprobe.scenarios.schema import Scenario

SESSION_SPACING_DAYS = 10.0

# $/Mtok (input, output). Local paths are $0; anthropic priced at Haiku-4.5 list.
PRICING_PER_MTOK = {"stub": (0.0, 0.0), "ollama": (0.0, 0.0), "anthropic": (1.0, 5.0)}


class AnchorValidationError(RuntimeError):
    """Raised (after persisting + rendering the VOID report) when anchors fail (ADR-0006)."""

    def __init__(self, report: anchors_mod.AnchorReport, run_dir: Path) -> None:
        super().__init__("; ".join(report.messages))
        self.report = report
        self.run_dir = run_dir


def _wire_contamination(suite: list[Scenario]) -> None:
    """Other users' current values for a probed key -> that probe's must_not_contain.

    The user's own expected value is excluded — with 20 users on 2-4 value vocabularies,
    someone else almost always shares your value, and flagging the CORRECT answer as
    contamination would poison every metric downstream.
    """
    current_by_key: dict[str, dict[str, str]] = {}
    for sc in suite:
        for probe in (p for s in sc.sessions for p in s.probes):
            if probe.fact_key not in current_by_key:
                current_by_key[probe.fact_key] = {}
            val = sc.current_value(probe.fact_key)
            if val is not None:
                current_by_key[probe.fact_key][sc.user_id] = val
    for sc in suite:
        for probe in (p for s in sc.sessions for p in s.probes):
            others = {
                v for uid, v in current_by_key.get(probe.fact_key, {}).items()
                if uid != sc.user_id
            }
            new = sorted(others - {probe.expected_value} - set(probe.must_not_contain))
            probe.must_not_contain.extend(new)


def _oracle_facts(scenario: Scenario) -> dict[str, str]:
    from memprobe.scenarios.schema import FactStatus

    return {f.key: f.value for f in scenario.facts if f.status == FactStatus.CURRENT}


def _provider(spec: str) -> str:
    return spec.partition(":")[0] if ":" in spec else spec


def _cost_usd(spec: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = PRICING_PER_MTOK.get(_provider(spec), (0.0, 0.0))
    return input_tokens / 1e6 * p_in + output_tokens / 1e6 * p_out


def _run_config_cell(
    mem_cfg: MemConfig, cfg: RunConfig, suites: dict[int, list[Scenario]]
) -> dict:
    """One matrix column: every seed, every user, every session. Returns the persisted row."""
    policy = resolve(mem_cfg)
    rows_success: list[float] = []
    rows_contam: list[float] = []
    rows_stale: list[float] = []
    tokens_in = tokens_out = embed_tokens = n_probes = 0

    for seed, suite in suites.items():
        store = LangGraphStore()  # fresh substrate per (config, seed): no cross-cell bleed
        model = build_model(cfg.models.agent)
        embedder = (
            build_embedder(cfg.models.embeddings) if policy.retrieval == "embedding" else None
        )
        agent = build_agent(policy, store, model, embedder=embedder)
        users = [sc.user_id for sc in suite]
        neighbor = {u: users[(i + 1) % len(users)] for i, u in enumerate(users)}
        responses: dict[str, dict[str, str]] = {u: {} for u in users}

        for session_index in range(cfg.scenarios.n_sessions):
            for sc in suite:  # lockstep: see module docstring
                out = agent.run_session(
                    sc.user_id,
                    sc.sessions[session_index],
                    now=session_index * SESSION_SPACING_DAYS,
                    memory_user_id=(
                        neighbor[sc.user_id] if policy.scramble_namespaces else None
                    ),
                    oracle_facts=_oracle_facts(sc) if policy.oracle else None,
                )
                for r in out:
                    responses[sc.user_id][r["probe_id"]] = r["response"]

        for sc in suite:  # one aggregate row per (user, seed) — the CI population
            probes = [p for s in sc.sessions for p in s.probes]
            scored = [score_probe(responses[sc.user_id].get(p.probe_id, ""), p) for p in probes]
            m = aggregate(scored)
            rows_success.append(m.task_success)
            rows_contam.append(m.contamination_rate)
            rows_stale.append(m.staleness_rate)
            n_probes += m.n_probes

        tokens_in += model.usage.input_tokens
        tokens_out += model.usage.output_tokens
        if agent.retriever is not None:
            embed_tokens += getattr(agent.retriever, "embed_tokens", 0)

    conf = cfg.report.confidence
    n_sessions_total = len(suites) * cfg.scenarios.n_users * cfg.scenarios.n_sessions
    return {
        "name": mem_cfg.name,
        "anchor": mem_cfg.anchor.value if mem_cfg.anchor else None,
        "task_success": asdict(mean_ci(rows_success, conf)),
        "contamination": asdict(mean_ci(rows_contam, conf)),
        "staleness": asdict(mean_ci(rows_stale, conf)),
        "tokens": {
            "input": tokens_in,
            "output": tokens_out,
            "embed": embed_tokens,
            "per_session": round((tokens_in + tokens_out) / max(1, n_sessions_total), 2),
        },
        "cost_usd": round(_cost_usd(cfg.models.agent, tokens_in, tokens_out), 4),
        "n_probes": n_probes,
    }


def _anchor_estimate(results: list[dict], anchor: Anchor):
    from memprobe.eval.stats import Estimate

    for row in results:
        if row["anchor"] == anchor.value:
            return Estimate(**row["task_success"])
    raise ValueError(
        f"config matrix is missing the mandatory {anchor.value} anchor (ADR-0006); "
        "every run must include memory_off, oracle, and shuffled_placebo configs"
    )


def run(config_path: str, *, out_root: str | Path = "data/runs") -> Path:
    """Execute the full matrix; persist + render; raise AnchorValidationError on a VOID run."""
    cfg = RunConfig.load(config_path)
    for anchor in Anchor:  # fail fast, before minutes of compute, not after
        if not any(c.anchor == anchor for c in cfg.configs):
            raise ValueError(
                f"config matrix is missing the mandatory {anchor.value} anchor (ADR-0006)"
            )

    suites: dict[int, list[Scenario]] = {}
    chance_values: list[float] = []
    for seed in cfg.scenarios.seeds:
        suite = generate_suite(seed, cfg.scenarios)
        _wire_contamination(suite)
        suites[seed] = suite
        chance_values += [
            chance_rate(sc, [p for s in sc.sessions for p in s.probes]) for sc in suite
        ]
    chance_floor = sum(chance_values) / len(chance_values) if chance_values else 0.0

    results = [_run_config_cell(mem_cfg, cfg, suites) for mem_cfg in cfg.configs]

    anchor_report = anchors_mod.validate(
        memory_off=_anchor_estimate(results, Anchor.CALIBRATION),
        oracle=_anchor_estimate(results, Anchor.UPPER_BOUND),
        placebo=_anchor_estimate(results, Anchor.PLACEBO),
        chance_floor=chance_floor,
    )

    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "config_path": str(config_path),
        "models": cfg.models.model_dump(),
        "scenarios": cfg.scenarios.model_dump(),
        "report": cfg.report.model_dump(),
        "chance_floor": chance_floor,
        "anchors": {"ok": anchor_report.ok, "messages": anchor_report.messages},
        "configs": results,
    }
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2))
    shutil.copy(config_path, run_dir / "config.yaml")
    latest = Path(out_root) / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_id)

    from memprobe.report.render import render

    render(run_dir)

    if not anchor_report.ok:  # ADR-0006: a run whose anchors fail is VOID — abort loudly
        raise AnchorValidationError(anchor_report, run_dir)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the memprobe ablation matrix.")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    try:
        out = run(args.config)
    except AnchorValidationError as e:
        print("RUN VOID — anchors failed:")
        for msg in e.report.messages:
            print(f"  {msg}")
        print(f"void run persisted to {e.run_dir}")
        sys.exit(2)
    print(f"run written to {out}")


if __name__ == "__main__":
    main()
