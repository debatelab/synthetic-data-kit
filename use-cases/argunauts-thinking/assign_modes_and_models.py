import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign each example in a raw subset JSON file to a "
            "(mode, model) combination and write one JSON per combination."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input raw subset JSON (list of records).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        required=True,
        help="List of mode identifiers (e.g. a b).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help=(
            "List of model identifiers to embed into filenames and metadata "
            "(e.g. kit.gpt-oss-120b kit.mixtral-8x22b-instruct ...)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where per-(mode,model) JSON files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible assignments.",
    )
    return parser.parse_args()


def balanced_assign(
    records: List[Dict], modes: List[str], models: List[str]
) -> Dict[str, List[Dict]]:
    """Assign records as evenly as possible across all (mode, model) combos.

    We implement a simple round-robin over the Cartesian product of modes and
    models. This keeps the implementation straightforward while ensuring an
    approximately balanced distribution.
    """

    if not records:
        return {}

    combos: List[Dict[str, str]] = []
    for mode in modes:
        for model in models:
            combos.append({"mode": mode, "model": model})

    assigned: Dict[str, List[Dict]] = {f"{c['mode']}::{c['model']}": [] for c in combos}

    # Shuffle once for fairness so that ordering in the input file does not
    # determine which examples go to which combination.
    indices = list(range(len(records)))
    random.shuffle(indices)

    combo_count = len(combos)
    for i, idx in enumerate(indices):
        combo = combos[i % combo_count]
        key = f"{combo['mode']}::{combo['model']}"
        ex = dict(records[idx])  # shallow copy so we can annotate
        ex.setdefault("mode", combo["mode"])
        ex.setdefault("model", combo["model"])
        assigned[key].append(ex)

    return assigned


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if args.seed is not None:
        random.seed(args.seed)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list of records.")

    assigned = balanced_assign(records, args.modes, args.models)

    # Derive a base stem from the input filename for clearer outputs
    # Example: deepa2-aaac01-thinking_train_raw.json -> deepa2-aaac01-thinking_train
    stem = input_path.stem
    if stem.endswith("_raw"):
        base = stem[:-4]
    else:
        base = stem

    output_dir.mkdir(parents=True, exist_ok=True)

    for key, group in assigned.items():
        mode, model = key.split("::", maxsplit=1)
        safe_model = model.replace("/", "-")
        out_path = output_dir / f"{base}_mode-{mode}_model-{safe_model}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(group, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
