"""Harness orchestration (milestone 5): anchors gate the run, metrics land where designed.

One tiny hermetic matrix run is shared by the whole module (module-scoped fixture) — the
assertions here are about ORCHESTRATION (wiring, gating, persistence, ordering), all of
which the small suite exercises fully.
"""

import json
from pathlib import Path

import pytest
import yaml

from memprobe.harness.run import AnchorValidationError, run

TINY = {
    "scenarios": {
        "n_users": 6, "n_sessions": 5, "turns_per_session": 4,
        "contradiction_rate": 0.5, "distractor_density": 0.3, "seeds": [0, 1],
    },
    "models": {"agent": "stub", "judge": "stub", "embeddings": "hash"},
    "configs": [
        {"name": "memory_off", "anchor": "calibration", "memory": "none"},
        {"name": "oracle", "anchor": "upper_bound", "memory": "oracle"},
        {"name": "shuffled_placebo", "anchor": "placebo", "memory": "semantic",
         "retrieval": "direct", "scramble_namespaces": True},
        {"name": "episodic_only", "memory": "episodic", "retrieval": "direct"},
        {"name": "semantic_gate_07", "memory": ["episodic", "semantic"],
         "retrieval": "direct", "write_gate_tau": 0.7},
        {"name": "semantic_gate_09", "memory": ["episodic", "semantic"],
         "retrieval": "direct", "write_gate_tau": 0.9},
        {"name": "semantic_embedding", "memory": ["episodic", "semantic"],
         "retrieval": "embedding", "write_gate_tau": 0.7},
        {"name": "semantic_forget_decay", "memory": ["episodic", "semantic"],
         "retrieval": "direct", "write_gate_tau": 0.7,
         "forgetting": "recency_decay", "half_life_days": 30},
    ],
    "report": {"confidence": 0.95, "out_dir": "report/out"},
}


def write_tiny_config(tmp_path: Path, **overrides) -> Path:
    cfg = json.loads(json.dumps(TINY))  # deep copy
    cfg.update(overrides)
    cfg["report"] = {**cfg["report"], "out_dir": str(tmp_path / "report_out")}
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny_run")
    run_dir = run(str(write_tiny_config(tmp)), out_root=tmp / "runs")
    return run_dir, json.loads((run_dir / "run.json").read_text())


def _row(data, name):
    return next(r for r in data["configs"] if r["name"] == name)


def test_anchors_hold_on_the_hermetic_matrix(tiny_run):
    _, data = tiny_run
    assert data["anchors"]["ok"], data["anchors"]["messages"]


def test_anchor_rows_land_where_designed(tiny_run):
    _, data = tiny_run
    # Refusing stub floor: exactly zero. Oracle ceiling: exactly one. Placebo: at most
    # coincidence (chance + tolerance), never separated above it.
    assert _row(data, "memory_off")["task_success"]["mean"] == 0.0
    assert _row(data, "oracle")["task_success"]["mean"] == 1.0
    placebo = _row(data, "shuffled_placebo")["task_success"]
    assert placebo["mean"] <= data["chance_floor"] + 0.05


def test_oracle_is_clean_of_contamination_and_staleness(tiny_run):
    _, data = tiny_run
    oracle = _row(data, "oracle")
    assert oracle["contamination"]["mean"] == 0.0
    assert oracle["staleness"]["mean"] == 0.0


def test_memory_configs_beat_the_floor(tiny_run):
    _, data = tiny_run
    floor = _row(data, "memory_off")["task_success"]["mean"]
    for name in ("episodic_only", "semantic_gate_07", "semantic_embedding"):
        assert _row(data, name)["task_success"]["mean"] > floor, name


def test_tau_09_is_more_conservative_than_07(tiny_run):
    """ADR-0008's designed fork at matrix scale: the strict gate refuses revisions (more
    stale answers) and refuses off-vocab junk (it cannot be MORE contaminated)."""
    _, data = tiny_run
    g07, g09 = _row(data, "semantic_gate_07"), _row(data, "semantic_gate_09")
    assert g09["staleness"]["mean"] >= g07["staleness"]["mean"]
    assert g09["contamination"]["mean"] <= g07["contamination"]["mean"]


def test_token_accounting_orders_by_work_done(tiny_run):
    _, data = tiny_run
    # memory_off answers probes only; adding extraction+summaries must cost more.
    assert (
        _row(data, "semantic_gate_07")["tokens"]["per_session"]
        > _row(data, "memory_off")["tokens"]["per_session"]
    )
    # only the embedding config meters encoder work
    assert _row(data, "semantic_embedding")["tokens"]["embed"] > 0
    assert _row(data, "semantic_gate_07")["tokens"]["embed"] == 0
    # stub runs are free
    assert all(row["cost_usd"] == 0.0 for row in data["configs"])


def test_run_dir_persists_everything_and_updates_latest(tiny_run):
    run_dir, data = tiny_run
    assert (run_dir / "run.json").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "results.md").exists()
    latest = run_dir.parent / "latest"
    assert latest.is_symlink() and latest.resolve() == run_dir.resolve()
    out_dir = Path(data["report"]["out_dir"])
    assert (out_dir / "results.md").exists()


def test_report_leads_with_anchors_and_labels_stub_numbers(tiny_run):
    run_dir, _ = tiny_run
    text = (run_dir / "results.md").read_text()
    assert "Anchors (credibility gates)" in text
    assert text.index("Anchors") < text.index("Ablation matrix")
    assert "Stub-model run" in text
    assert "Pareto" in text
    assert "[" in text and "]" in text  # intervals, not bare points


def test_missing_anchor_config_fails_fast(tmp_path):
    cfg = json.loads(json.dumps(TINY))
    cfg["configs"] = [c for c in cfg["configs"] if c.get("anchor") != "placebo"]
    cfg["report"]["out_dir"] = str(tmp_path / "out")
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="placebo"):
        run(str(path), out_root=tmp_path / "runs")


def test_failed_anchors_persist_void_report_and_raise(tmp_path, monkeypatch):
    from memprobe.eval import anchors as anchors_mod

    def sabotage(**kwargs):
        return anchors_mod.AnchorReport(ok=False, messages=["CALIBRATION FAIL: forced by test"])

    monkeypatch.setattr("memprobe.harness.run.anchors_mod.validate", sabotage)
    with pytest.raises(AnchorValidationError) as exc:
        run(str(write_tiny_config(tmp_path)), out_root=tmp_path / "runs")
    void_dir = exc.value.run_dir
    text = (void_dir / "results.md").read_text()
    assert "RUN VOID" in text
    assert "Ablation matrix" not in text  # findings suppressed, not decorated
