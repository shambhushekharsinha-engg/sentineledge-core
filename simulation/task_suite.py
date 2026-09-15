"""
task_suite.py -- Task suite definitions for the SentinelEdge challenge.

Defines multiple bimanual manipulation task variations, each with a
natural-language instruction, target objects, success threshold, and
description.

Usage (CLI)
-----------
python simulation/task_suite.py              # list all tasks
python simulation/task_suite.py --task set_table
python simulation/task_suite.py --verbose
python simulation/task_suite.py --json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_SUITE: Dict[str, Dict[str, Any]] = {
    "set_table": {
        "instruction": "Place the plate and cup on the table in their correct positions.",
        "primary_target": "plate",
        "secondary_target": "cup",
        "success_threshold": 0.25,
        "description": "Bimanual table setting: coordinate both arms to position dinnerware.",
    },
    "retrieve_spoon": {
        "instruction": "Open the drawer and retrieve the spoon, then place it beside the plate.",
        "primary_target": "spoon",
        "secondary_target": "drawer",
        "success_threshold": 0.20,
        "description": "Open drawer with one arm, retrieve spoon with the other.",
    },
    "handoff": {
        "instruction": "Pick up the cup with arm A, pass it to arm B, then place it on the table.",
        "primary_target": "cup",
        "secondary_target": "cup",
        "success_threshold": 0.15,
        "description": "Cross-arm handoff demonstrating bimanual coordination.",
    },
    "collision_avoid": {
        "instruction": (
            "Place the fork on the right side of the plate "
            "while avoiding the moving obstacle."
        ),
        "primary_target": "fork",
        "secondary_target": "dynamic_obstacle",
        "success_threshold": 0.20,
        "description": "Dynamic collision avoidance while completing manipulation.",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_task(name: str) -> Dict[str, Any]:
    """
    Return the task config dict for `name`.

    Raises
    ------
    ValueError
        If `name` is not a registered task.
    """
    if name not in TASK_SUITE:
        raise ValueError(
            f"Unknown task: '{name}'. "
            f"Available tasks: {list(TASK_SUITE.keys())}"
        )
    return TASK_SUITE[name]


def list_tasks(verbose: bool = False) -> None:
    """Print all registered tasks to stdout."""
    print(f"\n{'='*60}")
    print("  SentinelEdge Task Suite")
    print(f"{'='*60}")
    for name, cfg in TASK_SUITE.items():
        print(f"\n  [{name}]")
        print(f"    description : {cfg['description']}")
        if verbose:
            print(f"    instruction : {cfg['instruction']}")
            print(f"    primary     : {cfg['primary_target']}")
            print(f"    secondary   : {cfg['secondary_target']}")
            print(f"    threshold   : {cfg['success_threshold']}")
    print()


def task_names():
    """Return a list of all registered task names."""
    return list(TASK_SUITE.keys())


def to_json(indent: int = 2) -> str:
    """Return the full task suite as a formatted JSON string."""
    return json.dumps(TASK_SUITE, indent=indent)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="List or inspect SentinelEdge task definitions."
    )
    parser.add_argument("--task", type=str, default=None,
                        help="Show details for a specific task by name.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all fields when listing tasks.")
    parser.add_argument("--json", action="store_true",
                        help="Dump the task suite (or a single task) as JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.task:
        try:
            cfg = get_task(args.task)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            raise SystemExit(1) from exc

        if args.json:
            print(json.dumps({args.task: cfg}, indent=2))
        else:
            print(f"\n  Task : {args.task}")
            for k, v in cfg.items():
                print(f"    {k:<20}: {v}")
            print()
    else:
        if args.json:
            print(to_json())
        else:
            list_tasks(verbose=args.verbose)
