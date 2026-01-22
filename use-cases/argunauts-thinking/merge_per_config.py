import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge aligned per-(mode, model) JSON files for a given "
            "(config, split) into a single aligned split JSON."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Original dataset config name (e.g. deepa2-aaac01-thinking).",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Split name (e.g. train, validation, test).",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Root directory where aligned per-group outputs live.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the merged aligned JSON for this (config, split).",
    )
    return parser.parse_args()


def discover_group_files(input_root: Path, config: str, split: str) -> List[Path]:
    """Discover all aligned JSON files for a given (config, split).

    Expected layout: input_root/config/split/mode-*_model-*/<file>.json
    We keep it simple and include all .json files found under that directory.
    """

    base_dir = input_root / config / split
    if not base_dir.exists():
        raise FileNotFoundError(f"Aligned directory not found for {config}/{split}: {base_dir}")

    return sorted(base_dir.rglob("*.json"))


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Aligned file must contain a list of records: {path}")
    return data


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_path = Path(args.output)

    group_files = discover_group_files(input_root, args.config, args.split)
    if not group_files:
        raise FileNotFoundError(
            f"No aligned JSON files found for config={args.config}, split={args.split}."
        )

    all_records: List[Dict] = []
    seen_ids = set()

    for path in group_files:
        records = load_records(path)
        for ex in records:
            ex_id = ex.get("example_id")
            if ex_id is not None and ex_id in seen_ids:
                raise ValueError(
                    f"Duplicate example_id {ex_id!r} encountered when merging {path}. "
                    "This suggests overlapping assignments across groups."
                )
            if ex_id is not None:
                seen_ids.add(ex_id)
            all_records.append(ex)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
