#!/usr/bin/env python3
"""Extract action column from parquet files and write numbered JSON lines files.

Each parquet produces one JSON file, where each row in the action column is
written as a single JSON line. Output files are named 0.json, 1.json, ...
Ordered by the last three digits of the episode index in the filename.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


EPISODE_RE = re.compile(r"episode_(\d+)", re.IGNORECASE)


def _extract_episode_index(path: Path) -> int:
    match = EPISODE_RE.search(path.stem)
    if not match:
        return -1
    return int(match.group(1))


def _iter_parquet_files(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("*.parquet"))


def _write_actions_json(df: pd.DataFrame, out_path: Path) -> None:
    if "action" not in df.columns:
        raise ValueError(f"Missing 'action' column in {out_path}")

    with open(out_path, "w") as f:
        for val in df["action"]:
            if hasattr(val, "tolist"):
                val = val.tolist()
            f.write(json.dumps(val) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract action column from parquet files to numbered JSON files."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing parquet files")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output folder for JSON files (default: input_dir)",
    )
    parser.add_argument(
        "--pose_dir",
        default=None,
        help="Optional folder to search for pose.json and copy it to output_dir",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input_dir not found or not a directory: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(_iter_parquet_files(input_dir))
    if not parquet_files:
        raise ValueError(f"No parquet files found under: {input_dir}")

    def sort_key(p: Path) -> tuple[int, int, str]:
        episode_idx = _extract_episode_index(p)
        last_three = episode_idx % 1000 if episode_idx >= 0 else -1
        return (last_three, episode_idx, p.name)

    parquet_files.sort(key=sort_key)

    for i, parquet_path in enumerate(parquet_files):
        df = pd.read_parquet(parquet_path)
        out_path = output_dir / f"{i}.json"
        _write_actions_json(df, out_path)

    if args.pose_dir:
        pose_dir = Path(args.pose_dir)
        if not pose_dir.exists() or not pose_dir.is_dir():
            raise ValueError(f"pose_dir not found or not a directory: {pose_dir}")

        pose_src = pose_dir / "pose.jsonl"
        if pose_src.exists() and pose_src.is_file():
            shutil.copy2(pose_src, output_dir / "pose.jsonl")
    
    input_dir = Path(args.input_dir)
    if input_dir.exists() and input_dir.is_dir():
        shutil.rmtree(input_dir)

    return 0


if __name__ == "__main__":
    exit_code = main()

    
