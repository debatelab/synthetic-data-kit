import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish aligned deep-argmap configurations to Hugging Face as a multi-config dataset."
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
        required=True,
        help="Target dataset repo name on Hugging Face.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="List of original config names (e.g. deepa2-aaac01-thinking).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to publish per config.",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default="data/merged",
        help=(
            "Directory containing merged per-config split JSON files, as "
            "produced by run_alignment.sh (e.g. deepa2-aaac01-thinking-aligned_train.json)."
        ),
    )
    parser.add_argument(
        "--public",
        action="store_false",
        dest="private",
        help=(
            "Publish to a public dataset repo. By default, a new repo is "
            "created as private when first pushed."
        ),
    )
    parser.set_defaults(private=True)
    return parser.parse_args()


def load_split(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Merged file must contain a list of records: {path}")
    return data


def main() -> None:
    args = parse_args()

    merged_dir = Path(args.merged_dir)
    repo_id = f"{args.org}/{args.repo_name}"

    for config in args.configs:
        dd = DatasetDict()
        for split in args.splits:
            filename = f"{config}-aligned_{split}.json"
            path = merged_dir / filename
            if not path.exists():
                # It is valid for some splits to be missing
                continue
            records = load_split(path)
            dd[split] = Dataset.from_list(records)

        if not dd:
            print(f"No splits found for config={config}, skipping push.")
            continue

        config_name = f"{config}-aligned"
        print(
            f"Pushing config {config_name!r} to {repo_id} with splits: {list(dd.keys())} "
            f"(private={args.private})"
        )
        dd.push_to_hub(repo_id, config_name=config_name, private=args.private)


if __name__ == "__main__":
    main()
