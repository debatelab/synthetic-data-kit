#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
REPAIRED_DIR = Path("data/repaired")


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at top level: {path}")
    return data


def index_by_example_id(records: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    index: Dict[Any, Dict[str, Any]] = {}
    for rec in records:
        eid = rec.get("example_id")
        if eid is None:
            raise ValueError(f"Record missing 'example_id': {rec}")
        if eid in index:
            raise ValueError(f"Duplicate example_id {eid}")
        index[eid] = rec
    return index


def get_messages(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of messages for a record.

    Internal files use the key 'conversations'. If a record uses 'messages'
    instead, we fall back to that.
    """
    if "conversations" in rec:
        return rec["conversations"]
    if "messages" in rec:
        return rec["messages"]
    return []


def try_parse_json(s: Any) -> bool:
    """Return True if s can be parsed as JSON, False otherwise.

    Non-string values are treated as already-structured (and thus acceptable)
    and return True. This function is only used for diagnostics and repair
    decisions.
    """
    if not isinstance(s, str):
        return True
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def repair_record(
    raw_rec: Dict[str, Any], cleaned_rec: Dict[str, Any]
) -> Tuple[Dict[str, Any], bool]:
    """Attempt to repair tool messages in a single example.

    For each tool message, if the cleaned content is not valid JSON but the
    raw content is, we copy the raw content (and name) into the cleaned
    message. If both sides are non-JSON, the example is marked as
    unrepairable and the caller should skip it.
    """
    # Deep copy cleaned_rec so we don't mutate the input
    repaired: Dict[str, Any] = json.loads(json.dumps(cleaned_rec))

    conv_raw = get_messages(raw_rec)
    conv_rep = get_messages(repaired)

    if len(conv_raw) != len(conv_rep):
        # This should not happen for cleaned files; treat as unrepairable.
        return repaired, False

    for mr, mc in zip(conv_raw, conv_rep):
        if mr.get("role") == "tool" and mc.get("role") == "tool":
            raw_content = mr.get("content")
            cleaned_content = mc.get("content")

            raw_ok = try_parse_json(raw_content)
            cleaned_ok = try_parse_json(cleaned_content)

            # Cleaned tool content is fine: nothing to do.
            if cleaned_ok:
                continue

            # Raw is OK, cleaned is not: repair by copying raw fields.
            if raw_ok:
                mc["content"] = raw_content
                # Align tool name as well, just in case.
                mc["name"] = mr.get("name")
                continue

            # Both raw and cleaned tool contents are non-JSON.
            return repaired, False

    # Final sanity check: ensure all tool messages in repaired are JSON-parseable.
    for m in conv_rep:
        if m.get("role") == "tool" and not try_parse_json(m.get("content")):
            return repaired, False

    return repaired, True


def find_file_pairs() -> List[Tuple[Path, Path, str, str]]:
    """Return list of (raw_path, cleaned_path, config, split)."""
    pairs: List[Tuple[Path, Path, str, str]] = []
    for raw_path in sorted(RAW_DIR.glob("deepa2-*-thinking_*_raw.json")):
        name = raw_path.name
        # Example: deepa2-aaac01-thinking_train_raw.json
        base = name[: -len("_raw.json")]  # deepa2-aaac01-thinking_train
        if "_" not in base:
            continue
        config, split = base.rsplit("_", 1)
        cleaned_name = f"{config}-aligned_{split}.json"
        cleaned_path = CLEANED_DIR / cleaned_name
        if cleaned_path.exists():
            pairs.append((raw_path, cleaned_path, config, split))
        else:
            print(f"[WARN] No cleaned file for {raw_path} (expected {cleaned_path})")
    return pairs


def main() -> None:
    REPAIRED_DIR.mkdir(parents=True, exist_ok=True)

    pairs = find_file_pairs()
    if not pairs:
        print("No raw/cleaned file pairs found.")
        return

    for raw_path, cleaned_path, config, split in pairs:
        print(f"\n=== Repairing config: {config} | Split: {split} ===")
        print(f"Raw:     {raw_path}")
        print(f"Cleaned: {cleaned_path}")

        raw_records = load_json(raw_path)
        cleaned_records = load_json(cleaned_path)

        raw_index = index_by_example_id(raw_records)

        repaired_records: List[Dict[str, Any]] = []
        skipped_no_raw: List[Any] = []
        skipped_unrepairable: List[Any] = []

        for cleaned_rec in cleaned_records:
            eid = cleaned_rec.get("example_id")
            if eid is None:
                # Should not happen, but skip defensively.
                skipped_unrepairable.append(eid)
                continue

            raw_rec = raw_index.get(eid)

            # If we don't have a raw reference, keep the example only if all
            # tool messages in the cleaned record are JSON-parseable.
            if raw_rec is None:
                conv = get_messages(cleaned_rec)
                all_ok = True
                for m in conv:
                    if m.get("role") == "tool" and not try_parse_json(m.get("content")):
                        all_ok = False
                        break
                if all_ok:
                    repaired_records.append(cleaned_rec)
                else:
                    skipped_no_raw.append(eid)
                continue

            repaired, ok = repair_record(raw_rec, cleaned_rec)
            if ok:
                repaired_records.append(repaired)
            else:
                skipped_unrepairable.append(eid)

        out_path = REPAIRED_DIR / cleaned_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(repaired_records, f, ensure_ascii=False, indent=2)

        # Quick sanity count and report
        print(f"Total cleaned examples:     {len(cleaned_records)}")
        print(f"Total repaired examples:    {len(repaired_records)}")
        print(f"Skipped (no raw reference): {len(skipped_no_raw)}")
        print(f"Skipped (unrepairable):     {len(skipped_unrepairable)}")
        if skipped_no_raw:
            print(
                f"  example_ids (no raw):       {skipped_no_raw[:5]}{' ...' if len(skipped_no_raw) > 5 else ''}"
            )
        if skipped_unrepairable:
            print(
                f"  example_ids (unrepairable): {skipped_unrepairable[:5]}{' ...' if len(skipped_unrepairable) > 5 else ''}"
            )


if __name__ == "__main__":
    main()
