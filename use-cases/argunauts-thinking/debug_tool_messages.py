#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/repaired")

# Limit how many concrete diffs we print to keep output manageable
MAX_DIFFS_PER_FILE = 0
CONTENT_SNIPPET_LEN = 200


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at top level: {path}")
    return data


def index_by_example_id(records: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    index = {}
    for rec in records:
        eid = rec.get("example_id")
        if eid is None:
            raise ValueError("Record missing 'example_id': {}".format(rec))
        if eid in index:
            raise ValueError(f"Duplicate example_id {eid}")
        index[eid] = rec
    return index


def get_messages(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Internal files should use "conversations"; HF export might use "messages"
    if "conversations" in rec:
        return rec["conversations"]
    if "messages" in rec:
        return rec["messages"]
    return []


def snippet(s: Any, n: int = CONTENT_SNIPPET_LEN) -> str:
    if s is None:
        return "None"
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + "... [truncated]"


def try_parse_json(s: Any) -> bool:
    """Return True if s is a string that can be parsed as JSON, False otherwise.

    Non-string values are treated as already-structured (and thus "parseable")
    and return True. This function is only used for diagnostics.
    """
    if not isinstance(s, str):
        return True
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def compare_example(
    eid: Any,
    raw_rec: Dict[str, Any],
    cleaned_rec: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare tool messages and tool_calls for a single example.

    Returns a dict with:
      - changed_tool_messages: list of (msg_idx, raw_msg, cleaned_msg)
      - changed_tool_calls: list of (msg_idx, raw_tool_calls, cleaned_tool_calls)
      - length_mismatch: (len_raw, len_cleaned) or None
    """
    conv_raw = get_messages(raw_rec)
    conv_clean = get_messages(cleaned_rec)

    diffs: Dict[str, Any] = {
        "changed_tool_messages": [],
        "changed_tool_calls": [],
        "length_mismatch": None,
        "raw_tool_json_ok": True,
        "cleaned_tool_json_ok": True,
    }

    if len(conv_raw) != len(conv_clean):
        diffs["length_mismatch"] = (len(conv_raw), len(conv_clean))

    # Compare up to the min length; if lengths differ, that's captured above
    for i in range(min(len(conv_raw), len(conv_clean))):
        mr = conv_raw[i]
        mc = conv_clean[i]

        # Compare tool messages
        if mr.get("role") == "tool" or mc.get("role") == "tool":
            # roles must match if invariants held; we still check explicitly
            if (
                mr.get("role") != mc.get("role")
                or mr.get("name") != mc.get("name")
                or mr.get("content") != mc.get("content")
            ):
                diffs["changed_tool_messages"].append((i, mr, mc))

            # Track JSON parseability for diagnostics
            if not try_parse_json(mr.get("content")):
                diffs["raw_tool_json_ok"] = False
            if not try_parse_json(mc.get("content")):
                diffs["cleaned_tool_json_ok"] = False

        # Compare tool_calls on assistant messages
        if mr.get("role") == "assistant" and mc.get("role") == "assistant":
            if mr.get("tool_calls") != mc.get("tool_calls"):
                diffs["changed_tool_calls"].append((i, mr.get("tool_calls"), mc.get("tool_calls")))

    return diffs


def find_file_pairs() -> List[Tuple[Path, Path, str, str]]:
    """Return list of (raw_path, cleaned_path, config, split)."""
    pairs = []
    for raw_path in sorted(RAW_DIR.glob("deepa2-*-thinking_*_raw.json")):
        name = raw_path.name
        # Example: deepa2-aaac01-thinking_train_raw.json
        #          deepa2-aaac01-thinking_validation_raw.json
        #          deepa2-aaac01-thinking_test_raw.json
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
    pairs = find_file_pairs()
    if not pairs:
        print("No raw/cleaned file pairs found.")
        return

    for raw_path, cleaned_path, config, split in pairs:
        print(f"\n=== Config: {config} | Split: {split} ===")
        print(f"Raw:     {raw_path}")
        print(f"Cleaned: {cleaned_path}")

        raw_records = load_json(raw_path)
        cleaned_records = load_json(cleaned_path)

        raw_index = index_by_example_id(raw_records)
        cleaned_index = index_by_example_id(cleaned_records)

        shared_ids = sorted(set(raw_index.keys()) & set(cleaned_index.keys()))
        only_raw = sorted(set(raw_index.keys()) - set(cleaned_index.keys()))
        only_cleaned = sorted(set(cleaned_index.keys()) - set(raw_index.keys()))

        print(f"Total raw examples:     {len(raw_records)}")
        print(f"Total cleaned examples: {len(cleaned_records)}")
        print(f"Shared example_ids:     {len(shared_ids)}")
        if only_raw:
            print(f"Example_ids only in raw:     {len(only_raw)} (e.g. {only_raw[:5]})")
        if only_cleaned:
            print(f"Example_ids only in cleaned: {len(only_cleaned)} (e.g. {only_cleaned[:5]})")

        examples_with_tool = 0
        examples_identical_tool = 0
        examples_changed_tool = 0
        examples_changed_calls = 0
        length_mismatches = 0
        raw_tool_json_broken = 0
        cleaned_tool_json_broken = 0

        printed_diffs = 0

        for eid in shared_ids:
            raw_rec = raw_index[eid]
            cleaned_rec = cleaned_index[eid]

            diffs = compare_example(eid, raw_rec, cleaned_rec)
            has_tool_msg = any(m.get("role") == "tool" for m in get_messages(raw_rec)) or any(
                m.get("role") == "tool" for m in get_messages(cleaned_rec)
            )

            if has_tool_msg:
                examples_with_tool += 1

            changed_tool = bool(diffs["changed_tool_messages"])
            changed_calls = bool(diffs["changed_tool_calls"])

            if diffs["length_mismatch"] is not None:
                length_mismatches += 1

            if not diffs["raw_tool_json_ok"]:
                raw_tool_json_broken += 1
            if not diffs["cleaned_tool_json_ok"]:
                cleaned_tool_json_broken += 1

            if (
                has_tool_msg
                and not changed_tool
                and not changed_calls
                and diffs["length_mismatch"] is None
            ):
                examples_identical_tool += 1

            if changed_tool:
                examples_changed_tool += 1
            if changed_calls:
                examples_changed_calls += 1

            # Print concrete differences for a limited number of examples
            if (
                changed_tool or changed_calls or diffs["length_mismatch"] is not None
            ) and printed_diffs < MAX_DIFFS_PER_FILE:
                print(f"\n--- example_id={eid} ---")
                if diffs["length_mismatch"] is not None:
                    lr, lc = diffs["length_mismatch"]
                    print(f"  [LENGTH] conversations length raw={lr}, cleaned={lc}")

                for idx, mr, mc in diffs["changed_tool_messages"]:
                    print(f"  [TOOL MSG] message index {idx}")
                    print(f"    raw.role={mr.get('role')}, cleaned.role={mc.get('role')}")
                    print(f"    raw.name={mr.get('name')}, cleaned.name={mc.get('name')}")
                    print("    raw.content:    ", snippet(mr.get("content")))
                    print("    cleaned.content:", snippet(mc.get("content")))

                for idx, tr, tc in diffs["changed_tool_calls"]:
                    print(f"  [TOOL CALLS] assistant message index {idx}")
                    print("    raw.tool_calls:    ", snippet(tr))
                    print("    cleaned.tool_calls:", snippet(tc))

                printed_diffs += 1

        print("\nSummary for this file pair:")
        print(f"  Examples with any tool messages:        {examples_with_tool}")
        print(f"  Examples with identical tool payloads:  {examples_identical_tool}")
        print(f"  Examples with changed tool messages:    {examples_changed_tool}")
        print(f"  Examples with changed tool_calls:       {examples_changed_calls}")
        print(f"  Examples with length mismatches:        {length_mismatches}")
        print(f"  Examples with non-JSON raw tool content:     {raw_tool_json_broken}")
        print(f"  Examples with non-JSON cleaned tool content: {cleaned_tool_json_broken}")
        if printed_diffs >= MAX_DIFFS_PER_FILE:
            print(f"  (diff output truncated at {MAX_DIFFS_PER_FILE} examples)")


if __name__ == "__main__":
    main()
