"""
metrics.py — Evaluation metrics module for SentinelEdge challenge.

Provides TaskMetrics dataclass and MetricsLogger for per-seed tracking,
rich-table summaries, JSON export, and CSV export.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskMetrics:
    """Per-seed evaluation result."""
    seed: int
    success: bool
    steps_to_success: int          # -1 if episode never succeeded
    mean_reward: float
    avg_dist_left: float           # average end-effector distance (left arm)
    avg_dist_right: float          # average end-effector distance (right arm)
    collision_detected: bool


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Collects per-seed TaskMetrics and produces summary reports."""

    def __init__(self) -> None:
        self._records: List[TaskMetrics] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def log(self, seed: int, metrics_dict: dict) -> TaskMetrics:
        """
        Create and store a TaskMetrics entry.

        Parameters
        ----------
        seed : int
            Random seed used for the episode.
        metrics_dict : dict
            Keys (all optional, fall back to sensible defaults):
              success, steps_to_success, mean_reward,
              avg_dist_left, avg_dist_right, collision_detected
        """
        tm = TaskMetrics(
            seed=seed,
            success=bool(metrics_dict.get("success", False)),
            steps_to_success=int(metrics_dict.get("steps_to_success", -1)),
            mean_reward=float(metrics_dict.get("mean_reward", 0.0)),
            avg_dist_left=float(metrics_dict.get("avg_dist_left", 0.0)),
            avg_dist_right=float(metrics_dict.get("avg_dist_right", 0.0)),
            collision_detected=bool(metrics_dict.get("collision_detected", False)),
        )
        self._records.append(tm)
        return tm

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate(self) -> Dict[str, float]:
        if not self._records:
            return {}
        n = len(self._records)
        successes = [r for r in self._records if r.success]
        success_rate = len(successes) / n
        mean_steps = (
            sum(r.steps_to_success for r in successes) / len(successes)
            if successes else float("nan")
        )
        mean_reward = sum(r.mean_reward for r in self._records) / n
        mean_dist_left = sum(r.avg_dist_left for r in self._records) / n
        mean_dist_right = sum(r.avg_dist_right for r in self._records) / n
        collision_rate = sum(r.collision_detected for r in self._records) / n
        return {
            "num_episodes": n,
            "success_rate": round(success_rate, 4),
            "mean_steps_to_success": round(mean_steps, 2),
            "mean_reward": round(mean_reward, 4),
            "mean_avg_dist_left": round(mean_dist_left, 4),
            "mean_avg_dist_right": round(mean_dist_right, 4),
            "collision_rate": round(collision_rate, 4),
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a rich table of per-seed metrics plus aggregate row."""
        try:
            from rich.console import Console
            from rich.table import Table
            _rich = True
        except ImportError:
            _rich = False

        agg = self._aggregate()

        if _rich:
            console = Console()
            table = Table(title="[bold cyan]SentinelEdge Evaluation Metrics[/bold cyan]",
                          show_header=True, header_style="bold magenta")
            cols = ["Seed", "Success", "Steps", "Mean Reward",
                    "AvgDist L", "AvgDist R", "Collision"]
            for c in cols:
                table.add_column(c, justify="right")

            for r in self._records:
                suc_str = "[green]V[/green]" if r.success else "[red]X[/red]"
                col_str = "[red]Yes[/red]" if r.collision_detected else "No"
                steps_str = str(r.steps_to_success) if r.steps_to_success >= 0 else "-"
                table.add_row(
                    str(r.seed),
                    suc_str,
                    steps_str,
                    f"{r.mean_reward:.4f}",
                    f"{r.avg_dist_left:.4f}",
                    f"{r.avg_dist_right:.4f}",
                    col_str,
                )

            table.add_section()
            table.add_row(
                "[bold]AGGREGATE[/bold]",
                f"[bold]{agg.get('success_rate', 0):.1%}[/bold]",
                f"[bold]{agg.get('mean_steps_to_success', float('nan')):.1f}[/bold]",
                f"[bold]{agg.get('mean_reward', 0):.4f}[/bold]",
                f"[bold]{agg.get('mean_avg_dist_left', 0):.4f}[/bold]",
                f"[bold]{agg.get('mean_avg_dist_right', 0):.4f}[/bold]",
                f"[bold]{agg.get('collision_rate', 0):.1%}[/bold]",
            )
            console.print(table)
        else:
            header = (
                f"{'Seed':>6}  {'Success':>8}  {'Steps':>6}  "
                f"{'MeanRew':>10}  {'DistL':>8}  {'DistR':>8}  {'Collision':>9}"
            )
            print("\n=== SentinelEdge Evaluation Metrics ===")
            print(header)
            print("-" * len(header))
            for r in self._records:
                steps_str = str(r.steps_to_success) if r.steps_to_success >= 0 else "  -"
                print(
                    f"{r.seed:>6}  {str(r.success):>8}  {steps_str:>6}  "
                    f"{r.mean_reward:>10.4f}  {r.avg_dist_left:>8.4f}  "
                    f"{r.avg_dist_right:>8.4f}  {str(r.collision_detected):>9}"
                )
            print("-" * len(header))
            print(f"  success_rate          : {agg.get('success_rate', 0):.1%}")
            print(f"  mean_steps_to_success : {agg.get('mean_steps_to_success', float('nan')):.2f}")
            print(f"  mean_reward           : {agg.get('mean_reward', 0):.4f}")
            print(f"  collision_rate        : {agg.get('collision_rate', 0):.1%}")

    def save_json(self, path) -> None:
        """Save full report (per-seed + aggregate) to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "aggregate": self._aggregate(),
            "per_seed": [asdict(r) for r in self._records],
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[MetricsLogger] JSON report saved -> {path}")

    def save_csv(self, path) -> None:
        """Save per-seed metrics to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._records:
            print("[MetricsLogger] No records to save.")
            return
        fieldnames = list(asdict(self._records[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._records:
                writer.writerow(asdict(r))
        print(f"[MetricsLogger] CSV saved -> {path}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    logger = MetricsLogger()
    seeds = [42, 7, 13, 99, 2024, 1, 8, 55, 100, 300]

    print("Simulating 10 evaluation episodes ...")
    for seed in seeds:
        random.seed(seed)
        success = random.random() > 0.4
        logger.log(seed, {
            "success": success,
            "steps_to_success": random.randint(50, 450) if success else -1,
            "mean_reward": random.uniform(0.3, 1.0),
            "avg_dist_left": random.uniform(0.05, 0.35),
            "avg_dist_right": random.uniform(0.05, 0.35),
            "collision_detected": random.random() < 0.15,
        })

    logger.summary()
    logger.save_json("results/metrics_report.json")
    logger.save_csv("results/metrics_report.csv")
