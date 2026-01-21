import argparse
from pathlib import Path
from typing import Dict, List
import random

from datasets import Dataset, DatasetDict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Mode A / Mode B aligned argunauts datasets "
            "and publish Mode A, Mode B, and merged configs to Hugging Face."
        )
    )
    parser.add_argument(
        "--org",
        type=str,
        required=True,
        help="Hugging Face organization/user name (e.g. 'YOUR_ORG').",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="argunauts-thinking-aligned",
        help="Target dataset repo name on Hugging Face.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for the merge policy. "
            "If set, the random choice between Mode A and Mode B "
            "will be reproducible."
        ),
    )
    return parser.parse_args()


def load_aligned_json(script_dir: Path, mode: str, split: str) -> List[Dict]:
    path = script_dir / f"aligned_mode_{mode}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Aligned file not found for mode {mode}, split {split}: {path}")
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.seed is not None:
        print(f"Seeding RNG with seed={args.seed} for merge policy")
        random.seed(args.seed)

    repo_id = f"{args.org}/{args.repo_name}"
    print(f"Target Hugging Face dataset: {repo_id}")

    dd_mode_a = DatasetDict()
    dd_mode_b = DatasetDict()
    dd_merged = DatasetDict()

    for split in args.splits:
        print(f"Processing split: {split}")

        records_a = load_aligned_json(script_dir, mode="a", split=split)
        records_b = load_aligned_json(script_dir, mode="b", split=split)

        if len(records_a) != len(records_b):
            raise ValueError(
                f"Split {split}: Mode A has {len(records_a)} examples, "
                f"Mode B has {len(records_b)} examples. Lengths must match for merging by position."
            )

        ds_a = Dataset.from_list(records_a)
        ds_b = Dataset.from_list(records_b)

        # Merge policy: randomly choose Mode A or Mode B per example.
        # This keeps exactly one aligned conversation per original example,
        # while mixing both alignment strategies across the dataset.
        merged_records = []
        for ex_a, ex_b in zip(records_a, records_b):
            choice = random.choice([ex_a, ex_b])
            merged_records.append(choice)

        ds_merged = Dataset.from_list(merged_records)

        dd_mode_a[split] = ds_a
        dd_mode_b[split] = ds_b
        dd_merged[split] = ds_merged

    print("Pushing Mode A config (mode-a)...")
    dd_mode_a.push_to_hub(repo_id, config_name="mode-a")

    print("Pushing Mode B config (mode-b)...")
    dd_mode_b.push_to_hub(repo_id, config_name="mode-b")

    print("Pushing merged config (merged)...")
    dd_merged.push_to_hub(repo_id, config_name="merged")

    print("Done.")


if __name__ == "__main__":
    main()
