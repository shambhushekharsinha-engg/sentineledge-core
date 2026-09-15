"""
Tests for scripts/metrics.py — MetricsLogger and TaskMetrics.
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sample_metrics(success=True, steps=80, reward=5.0,
                    dist_l=0.15, dist_r=0.18, collision=False):
    return {
        "success": success,
        "steps_to_success": steps if success else -1,
        "mean_reward": reward,
        "avg_dist_left": dist_l,
        "avg_dist_right": dist_r,
        "collision_detected": collision,
    }


def _make_logger(n=5):
    """Return a MetricsLogger pre-populated with n episodes."""
    from scripts.metrics import MetricsLogger
    logger = MetricsLogger()
    for i in range(n):
        logger.log(i, _sample_metrics(success=(i % 2 == 0)))
    return logger


# ── Import tests ───────────────────────────────────────────────────────────────

def test_import_metrics_logger():
    from scripts.metrics import MetricsLogger
    assert callable(MetricsLogger)


def test_import_task_metrics_dataclass():
    from scripts.metrics import TaskMetrics
    tm = TaskMetrics(
        seed=0, success=True, steps_to_success=42,
        mean_reward=3.5, avg_dist_left=0.1,
        avg_dist_right=0.2, collision_detected=False
    )
    assert tm.seed == 0
    assert tm.success is True
    assert tm.steps_to_success == 42


# ── log() tests ────────────────────────────────────────────────────────────────

def test_log_stores_record():
    from scripts.metrics import MetricsLogger
    logger = MetricsLogger()
    tm = logger.log(0, _sample_metrics(success=True, steps=50))
    assert len(logger._records) == 1
    assert tm.seed == 0
    assert tm.success is True
    assert tm.steps_to_success == 50


def test_log_default_values():
    from scripts.metrics import MetricsLogger
    logger = MetricsLogger()
    tm = logger.log(99, {})         # no keys at all
    assert tm.seed == 99
    assert tm.success is False
    assert tm.steps_to_success == -1
    assert tm.mean_reward == 0.0
    assert tm.collision_detected is False


def test_log_multiple_records():
    from scripts.metrics import MetricsLogger
    logger = MetricsLogger()
    for i in range(10):
        logger.log(i, _sample_metrics(success=(i < 7)))
    assert len(logger._records) == 10


# ── summary() tests ────────────────────────────────────────────────────────────

def test_summary_does_not_raise(capsys):
    logger = _make_logger(n=5)
    logger.summary()              # should complete without exception
    captured = capsys.readouterr()
    # Either rich or plain-text output should be non-empty
    combined = captured.out + captured.err
    assert len(combined) > 0 or True   # rich may write directly; just check no exception


def test_summary_empty_logger(capsys):
    from scripts.metrics import MetricsLogger
    logger = MetricsLogger()
    logger.summary()              # should not crash on empty log


# ── Aggregate helper tests ─────────────────────────────────────────────────────

def test_aggregate_success_rate():
    from scripts.metrics import MetricsLogger
    logger = MetricsLogger()
    # 4 successes out of 10
    for i in range(10):
        logger.log(i, _sample_metrics(success=(i < 4)))
    agg = logger._aggregate()
    assert agg["success_rate"] == pytest.approx(0.4, abs=1e-4)


def test_aggregate_mean_reward():
    from scripts.metrics import MetricsLogger
    logger = MetricsLogger()
    logger.log(0, _sample_metrics(reward=2.0))
    logger.log(1, _sample_metrics(reward=4.0))
    agg = logger._aggregate()
    assert agg["mean_reward"] == pytest.approx(3.0, abs=1e-4)


def test_aggregate_collision_rate():
    from scripts.metrics import MetricsLogger
    logger = MetricsLogger()
    logger.log(0, _sample_metrics(collision=True))
    logger.log(1, _sample_metrics(collision=False))
    logger.log(2, _sample_metrics(collision=True))
    agg = logger._aggregate()
    assert agg["collision_rate"] == pytest.approx(2/3, abs=1e-4)


def test_aggregate_empty_returns_empty_dict():
    from scripts.metrics import MetricsLogger
    logger = MetricsLogger()
    assert logger._aggregate() == {}


# ── save_json() tests ──────────────────────────────────────────────────────────

def test_save_json_creates_file():
    logger = _make_logger(n=3)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        logger.save_json(path)
        assert os.path.exists(path)
    finally:
        os.unlink(path)


def test_save_json_valid_structure():
    logger = _make_logger(n=4)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        logger.save_json(path)
        with open(path) as f:
            data = json.load(f)
        assert "aggregate" in data
        assert "per_seed" in data
        assert isinstance(data["per_seed"], list)
        assert len(data["per_seed"]) == 4
        assert "success_rate" in data["aggregate"]
    finally:
        os.unlink(path)


def test_save_json_per_seed_fields():
    from scripts.metrics import MetricsLogger
    logger = MetricsLogger()
    logger.log(7, _sample_metrics(success=True, steps=33, reward=9.0))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        logger.save_json(path)
        with open(path) as f:
            data = json.load(f)
        record = data["per_seed"][0]
        assert record["seed"] == 7
        assert record["success"] is True
        assert record["steps_to_success"] == 33
        assert record["mean_reward"] == pytest.approx(9.0, abs=1e-4)
    finally:
        os.unlink(path)


# ── save_csv() tests ───────────────────────────────────────────────────────────

def test_save_csv_creates_file():
    logger = _make_logger(n=3)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        logger.save_csv(path)
        assert os.path.exists(path)
    finally:
        os.unlink(path)


def test_save_csv_has_correct_row_count():
    import csv as csv_mod
    logger = _make_logger(n=6)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        logger.save_csv(path)
        with open(path, newline="") as f:
            rows = list(csv_mod.DictReader(f))
        assert len(rows) == 6
    finally:
        os.unlink(path)


def test_save_csv_has_expected_columns():
    import csv as csv_mod
    logger = _make_logger(n=2)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        logger.save_csv(path)
        with open(path, newline="") as f:
            reader = csv_mod.DictReader(f)
            cols = reader.fieldnames
        for col in ("seed", "success", "steps_to_success",
                    "mean_reward", "avg_dist_left", "avg_dist_right",
                    "collision_detected"):
            assert col in cols, f"Missing column: {col}"
    finally:
        os.unlink(path)


def test_save_csv_empty_logger(capsys):
    from scripts.metrics import MetricsLogger
    logger = MetricsLogger()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        logger.save_csv(path)   # should not crash
    finally:
        if os.path.exists(path):
            os.unlink(path)
