"""Render a run directory into the results-first report + accuracy/cost Pareto.

The report leads with the anchor block (calibration / oracle / placebo) so a reader sees the
credibility gates before any findings. Every metric prints as `mean [lo, hi]` — never a bare
point estimate (ADR-0006). If anchors failed, a prominent VOID banner replaces the findings.

Output: results.md (+ pareto.png when matplotlib is installed — the core stays light, the
ASCII Pareto in results.md is always there). Written to both the run directory (archival)
and the configured report out_dir (the "latest report" the README points at).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VOID_BANNER = (
    "# ⛔ RUN VOID — ANCHORS FAILED\n\n"
    "This run's credibility gates did not hold. **No findings are reported** (a number\n"
    "without its anchors is not a result — NOTES.md ADR-0006).\n"
)


def _fmt(est: dict) -> str:
    return f"{est['mean']:.3f} [{est['lo']:.3f}, {est['hi']:.3f}]"


def _anchor_block(data: dict) -> list[str]:
    ok = data["anchors"]["ok"]
    lines = ["## Anchors (credibility gates)", ""]
    lines.append(f"- chance floor (exact, from closed value sets): **{data['chance_floor']:.3f}**")
    by_anchor = {row["anchor"]: row for row in data["configs"] if row["anchor"]}
    label = {
        "calibration": "memory OFF (must not exceed chance + tolerance)",
        "upper_bound": "oracle (ground truth injected; the ceiling)",
        "placebo": "shuffled other-user memory (must not beat coincidence)",
    }
    for key in ("calibration", "upper_bound", "placebo"):
        row = by_anchor.get(key)
        if row:
            lines.append(f"- {label[key]}: task success {_fmt(row['task_success'])}")
    lines.append("")
    lines.append(f"**Gate result: {'PASS' if ok else 'FAIL'}**")
    lines += [f"- {m}" for m in data["anchors"]["messages"]]
    return lines


def _config_table(data: dict) -> list[str]:
    lines = [
        "## Ablation matrix",
        "",
        "| config | task success | contamination | stale answers | tokens/session | est. $/run |",
        "|---|---|---|---|---|---|",
    ]
    for row in data["configs"]:
        name = row["name"] + (f" ⚓{row['anchor']}" if row["anchor"] else "")
        embed = f" (+{row['tokens']['embed']} embed)" if row["tokens"]["embed"] else ""
        lines.append(
            f"| {name} | {_fmt(row['task_success'])} | {_fmt(row['contamination'])} "
            f"| {_fmt(row['staleness'])} | {row['tokens']['per_session']}{embed} "
            f"| ${row['cost_usd']:.4f} |"
        )
    return lines


def _pareto_points(data: dict) -> list[tuple[str, float, float]]:
    """(name, tokens/session incl. embedding share, task success) for STUDY configs only —
    anchors are gates, not contenders."""
    pts = []
    n_sessions = data["scenarios"]["n_sessions"] * data["scenarios"]["n_users"] * len(
        data["scenarios"]["seeds"]
    )
    for row in data["configs"]:
        if row["anchor"]:
            continue
        cost = row["tokens"]["per_session"] + row["tokens"]["embed"] / max(1, n_sessions)
        pts.append((row["name"], cost, row["task_success"]["mean"]))
    return pts


def _pareto_frontier(pts: list[tuple[str, float, float]]) -> list[str]:
    """Configs no other config beats on BOTH axes (higher success, lower cost)."""
    frontier = []
    for name, cost, acc in pts:
        dominated = any(
            (c2 <= cost and a2 >= acc) and (c2 < cost or a2 > acc)
            for n2, c2, a2 in pts if n2 != name
        )
        if not dominated:
            frontier.append(name)
    return frontier


def _ascii_pareto(pts: list[tuple[str, float, float]]) -> list[str]:
    if not pts:
        return []
    width, height = 44, 12
    min_c, max_c = min(p[1] for p in pts), max(p[1] for p in pts)
    span = (max_c - min_c) or 1.0
    grid = [[" "] * width for _ in range(height)]
    legend = []
    for i, (name, cost, acc) in enumerate(sorted(pts, key=lambda p: p[1])):
        mark = chr(ord("A") + i)
        col = int((cost - min_c) / span * (width - 1))
        row = int((1.0 - max(0.0, min(1.0, acc))) * (height - 1))
        grid[row][col] = mark
        legend.append(f"  {mark} = {name} (tokens/session {cost:.1f}, success {acc:.3f})")
    lines = ["```", "task success ↑ (1.0 top) vs tokens/session → (cheap left)", ""]
    lines += ["  |" + "".join(r) for r in grid]
    lines.append("  +" + "-" * width)
    lines += legend
    lines.append("```")
    return lines


def _png_pareto(pts: list[tuple[str, float, float]], out: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, cost, acc in pts:
        ax.scatter(cost, acc, s=60)
        ax.annotate(name, (cost, acc), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("tokens / session (incl. embedding share)")
    ax.set_ylabel("task success (mean)")
    ax.set_title("memprobe: accuracy vs cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def _judge_block(data: dict) -> list[str]:
    lines = ["## Response quality (LLM-judge, secondary)", ""]
    judge = data.get("judge")
    if not judge:
        spec = data["models"]["judge"]
        lines.append(
            f"Not run (judge model: `{spec}`). Judge numbers may only appear together with a "
            "hand-labeled agreement audit (ADR-0005); the deterministic stub cannot produce "
            "an auditable judgment, so this section is intentionally empty rather than "
            "decorative."
        )
        return lines
    audit = judge["audit"]
    lines.append(
        f"Agreement audit: {audit['agreement']:.2f} raw agreement, Cohen's κ "
        f"{audit['cohen_kappa']:.2f} on n={audit['n_sampled']} hand-labeled samples."
    )
    for name, q in judge["quality_by_config"].items():
        lines.append(f"- {name}: mean quality {q:.3f}")
    return lines


def render(run_dir: str | Path) -> Path:
    """Read data/runs/<run>/run.json and write results.md (+ pareto.png) — VOID-aware."""
    run_dir = Path(run_dir)
    data = json.loads((run_dir / "run.json").read_text())
    ok = data["anchors"]["ok"]
    out_dir = Path(data["report"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if not ok:
        lines.append(VOID_BANNER)
    lines += [
        f"# memprobe results — run `{data['run_id']}`" if ok else "",
        "",
        f"- agent model: `{data['models']['agent']}` · judge: `{data['models']['judge']}` · "
        f"embeddings: `{data['models']['embeddings']}`",
        f"- scenarios: {data['scenarios']['n_users']} users × {data['scenarios']['n_sessions']} "
        f"sessions × {data['scenarios']['turns_per_session']} turns; "
        f"contradiction {data['scenarios']['contradiction_rate']}, "
        f"distractors {data['scenarios']['distractor_density']}; "
        f"seeds {data['scenarios']['seeds']}",
        f"- every interval: mean [lo, hi] at {int(data['report']['confidence'] * 100)}% "
        f"confidence across users × seeds "
        f"(n = {data['scenarios']['n_users'] * len(data['scenarios']['seeds'])} per config)",
        "",
    ]
    if data["models"]["agent"] == "stub":
        lines += [
            "> **Stub-model run.** The agent is the deterministic stub (a harness driver, "
            "not a real model): these numbers validate the harness, anchors, and policy "
            "mechanics — treat absolute values as harness properties, not model quality.",
            "",
        ]
    lines += _anchor_block(data)
    lines.append("")
    if ok:
        lines += _config_table(data)
        lines.append("")
        pts = _pareto_points(data)
        lines += ["## Accuracy vs cost (Pareto)", ""]
        frontier = _pareto_frontier(pts)
        lines.append(f"Pareto-efficient study configs: **{', '.join(frontier) or 'none'}**")
        lines.append("")
        lines += _ascii_pareto(pts)
        if _png_pareto(pts, out_dir / "pareto.png"):
            lines += ["", "![accuracy vs cost](pareto.png)"]
        lines.append("")
        lines += _judge_block(data)
        lines += [
            "",
            "## Reading these numbers",
            "",
            "- Two configs differ only if their intervals separate (eval/stats.separated) — "
            "overlapping CIs are a tie, whatever the means say.",
            "- Contamination counts responses asserting a value with no support in the "
            "user's own history (other users' current values); the statistical cross-user "
            "leak gate is the placebo anchor (ADR-0013).",
            "- Stale answers are the flagged-AND-wrong subset of the same must_not_contain "
            "channel (superseded own values and other-user values share it in v1 — "
            "metrics.py says so out loud; a per-user unique-token tracer is the sharper v2 "
            "instrument). For the placebo column that makes 'stale' read as 'served a "
            "neighbor's value and missed'.",
            "- The oracle-vs-best-config gap includes same-session updates, which no memory "
            "policy can answer under the memory-only respond context (ADR-0010).",
        ]

    text = "\n".join(lines).strip() + "\n"
    (run_dir / "results.md").write_text(text)
    out_path = out_dir / "results.md"
    out_path.write_text(text)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a memprobe run report.")
    ap.add_argument("--run", default="data/runs/latest")
    args = ap.parse_args()
    print(f"report written to {render(args.run)}")


if __name__ == "__main__":
    main()
