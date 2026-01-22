import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that transformed conversations preserve the structural "
            "layout of the original ones (message count, keys, roles)."
        )
    )
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original JSON (list of {example_id, conversations}).",
    )
    parser.add_argument(
        "--transformed",
        type=str,
        required=True,
        help="Path to transformed JSON (list of {example_id, conversations}).",
    )
    parser.add_argument(
        "--strict-ids",
        action="store_true",
        help=("Treat extra/missing example_ids as structural errors during validation."),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "If set, write a cleaned transformed JSON containing only structurally "
            "valid examples (drop bad ones)."
        ),
    )
    parser.add_argument(
        "--clean-output",
        type=str,
        default=None,
        help=(
            "Optional path for the cleaned transformed JSON. "
            "If omitted, a *_cleaned.json file is written next to --transformed."
        ),
    )
    return parser.parse_args()


def load_as_dict(path: Path) -> Dict[Any, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Top-level JSON must be a list in {path}")

    records: Dict[Any, Dict[str, Any]] = {}
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Record {idx} in {path} is not an object")
        example_id = item.get("example_id", idx)
        if example_id in records:
            raise ValueError(f"Duplicate example_id {example_id!r} in {path}")
        records[example_id] = item

    return records


def compare_message_structures(
    orig_messages: List[Dict[str, Any]],
    transf_messages: List[Dict[str, Any]],
    example_id: Any,
) -> List[str]:
    errors: List[str] = []

    if len(orig_messages) != len(transf_messages):
        errors.append(
            f"[example_id={example_id}] different number of messages: "
            f"original={len(orig_messages)}, transformed={len(transf_messages)}"
        )
        # Still compare overlapping prefix to show more issues.

    max_len = min(len(orig_messages), len(transf_messages))

    for i in range(max_len):
        o = orig_messages[i]
        t = transf_messages[i]

        if not isinstance(o, dict) or not isinstance(t, dict):
            errors.append(
                f"[example_id={example_id}, msg#{i}] message is not an object "
                f"(original type={type(o)}, transformed type={type(t)})"
            )
            continue

        orig_keys = set(o.keys())
        transf_keys = set(t.keys())
        if orig_keys != transf_keys:
            missing = orig_keys - transf_keys
            extra = transf_keys - orig_keys
            errors.append(
                f"[example_id={example_id}, msg#{i}] message keys differ: "
                f"missing_in_transformed={sorted(missing)}, "
                f"extra_in_transformed={sorted(extra)}"
            )

        # Role must be identical.
        orig_role = o.get("role")
        transf_role = t.get("role")
        if orig_role != transf_role:
            errors.append(
                f"[example_id={example_id}, msg#{i}] role mismatch: "
                f"original={orig_role!r}, transformed={transf_role!r}"
            )

        # Optional stricter checks (can be enabled if desired):
        for field in ("name", "tool_calls", "tools"):
            if o.get(field) != t.get(field):
                errors.append(
                    f"[example_id={example_id}, msg#{i}] field {field!r} changed: "
                    f"original={o.get(field)!r}, transformed={t.get(field)!r}"
                )

    return errors


def validate_structures(
    original_path: Path,
    transformed_path: Path,
    strict_ids: bool = False,
) -> Tuple[
    bool,
    List[str],
    Dict[Any, Dict[str, Any]],
    Dict[Any, Dict[str, Any]],
    Dict[Any, List[str]],
]:
    orig_records = load_as_dict(original_path)
    transf_records = load_as_dict(transformed_path)

    errors: List[str] = []
    orig_ids = set(orig_records.keys())
    transf_ids = set(transf_records.keys())

    missing_in_transformed = orig_ids - transf_ids
    extra_in_transformed = transf_ids - orig_ids

    if missing_in_transformed:
        errors.append(
            "Missing examples in transformed: "
            + ", ".join(map(repr, sorted(missing_in_transformed)))
        )

    if extra_in_transformed:
        msg = "Extra examples in transformed (not present in original): " + ", ".join(
            map(repr, sorted(extra_in_transformed))
        )
        if strict_ids:
            errors.append(msg)
        else:
            # Just warn; not necessarily a structural error.
            print("WARNING:", msg)

    per_example_errors: Dict[Any, List[str]] = {}

    for example_id in sorted(orig_ids & transf_ids):
        orig_item = orig_records[example_id]
        transf_item = transf_records[example_id]

        orig_conv = orig_item.get("conversations")
        transf_conv = transf_item.get("conversations")

        if not isinstance(orig_conv, list) or not isinstance(transf_conv, list):
            msg = (
                f"[example_id={example_id}] 'conversations' is not a list "
                f"(original type={type(orig_conv)}, transformed type={type(transf_conv)})"
            )
            errors.append(msg)
            per_example_errors.setdefault(example_id, []).append(msg)
            continue

        msg_errors = compare_message_structures(orig_conv, transf_conv, example_id)
        if msg_errors:
            errors.extend(msg_errors)
            per_example_errors.setdefault(example_id, []).extend(msg_errors)

    ok = len(errors) == 0
    return ok, errors, orig_records, transf_records, per_example_errors


def main() -> None:
    args = parse_args()
    original_path = Path(args.original)
    transformed_path = Path(args.transformed)

    (
        ok,
        errors,
        orig_records,
        transf_records,
        per_example_errors,
    ) = validate_structures(
        original_path=original_path,
        transformed_path=transformed_path,
        strict_ids=args.strict_ids,
    )

    if not args.clean:
        if ok:
            print(
                "[validate] Structures match: message counts, keys, roles, and metadata fields are preserved."
            )
            return

        print("[validate] Structural mismatches found:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    # Clean mode: drop structurally bad examples and write cleaned JSON.
    # Determine output path.
    if args.clean_output:
        cleaned_path = Path(args.clean_output)
    else:
        transformed_str = str(transformed_path)
        if transformed_str.lower().endswith(".json"):
            cleaned_path = transformed_path.with_name(transformed_path.stem + "_cleaned.json")
        else:
            cleaned_path = transformed_path.with_name(transformed_path.name + "_cleaned")

    orig_ids = set(orig_records.keys())
    transf_ids = set(transf_records.keys())

    bad_ids = set(per_example_errors.keys())

    # If strict_ids is enabled, also treat missing/extra IDs as bad.
    if args.strict_ids:
        missing_in_transformed = orig_ids - transf_ids
        extra_in_transformed = transf_ids - orig_ids
        bad_ids.update(missing_in_transformed)
        bad_ids.update(extra_in_transformed)

    kept_records = []
    for ex_id, record in transf_records.items():
        if ex_id in bad_ids:
            continue
        kept_records.append(record)

    total = len(transf_records)
    kept = len(kept_records)
    dropped = total - kept

    with cleaned_path.open("w", encoding="utf-8") as f:
        json.dump(kept_records, f, ensure_ascii=False, indent=2)

    print(
        "[clean] Wrote cleaned transformed file:",
        cleaned_path,
        f"(total={total}, kept={kept}, dropped={dropped})",
    )

    # Exit code: 0 as long as we successfully wrote the cleaned file.
    # Callers can inspect the summary to decide how to treat dropped examples.


if __name__ == "__main__":
    main()
