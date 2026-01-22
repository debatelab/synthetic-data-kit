import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a JSON conversations file from a Hugging Face dataset "
            "(e.g. DebateLabKIT/argunauts-thinking or deep-argmap variants)."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DebateLabKIT/argunauts-thinking",
        help="Hugging Face dataset identifier to load.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help=(
            "Optional dataset configuration name (for multi-config datasets). "
            "If provided, passed as the 'name' argument to load_dataset."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split to load (e.g. 'train', 'validation', 'test').",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "Optional number of examples to take from the split. If omitted, use the full split."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=("Optional seed for rg."),
    )
    parser.add_argument(
        "--messages-field",
        type=str,
        default="messages",
        help=(
            "Name of the field that contains the list of conversation messages "
            "in each example (e.g. 'messages' or 'conversations')."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path. If omitted, a default name of the form "
            "argunauts_<split>_conversations.json will be used in this directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine script directory for sensible default paths
    script_dir = Path(__file__).resolve().parent

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = script_dir / f"argunauts_{args.split}_conversations.json"

    # Build split expression if n is provided
    split_expr = args.split
    if args.n is not None:
        if args.n <= 0:
            raise ValueError("--n must be positive if provided")
        split_expr = f"{args.split}[:{args.n}]"

    load_kwargs = {"split": split_expr}
    if args.config_name is not None:
        load_kwargs["name"] = args.config_name

    print(f"Loading dataset {args.dataset!r}, config {args.config_name!r}, split {split_expr!r}...")
    ds = load_dataset(args.dataset, **load_kwargs)
    ds = ds.shuffle(seed=args.seed)

    messages_field = args.messages_field
    if messages_field not in ds.column_names:
        raise ValueError(
            f"Field {messages_field!r} not found in dataset columns {ds.column_names}. "
            "Use --messages-field to point to the list of messages."
        )

    records = []
    for idx, ex in enumerate(ds):
        messages = ex[messages_field]
        # We also keep a simple example_id to help with potential downstream matching
        records.append(
            {
                "example_id": ex.get("id", idx),
                "conversations": messages,
            }
        )

    print(f"Writing {len(records)} examples to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
