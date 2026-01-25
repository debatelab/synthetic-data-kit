from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv

DEFAULT_CONFIGS: List[str] = [
    "deepa2-aaac01-thinking",
    "deepa2-aaac02-thinking",
    "deepa2-aaac03-thinking",
    "deepa2-folly-thinking",
]

DEFAULT_SPLITS: List[str] = ["train", "validation", "test"]
DEFAULT_MODES: List[str] = ["a", "b"]
DEFAULT_MODELS: List[str] = [
    "kit.gpt-oss-120b",
    "kit.qwen3-vl-235b-a22b-instruct",
]

DEFAULT_BASE_DATASET_ID = "DebateLabKIT/argunauts-thinking"
DEFAULT_MAX_REPAIR_LOOPS = 3


@dataclass
class OrchestratorConfig:
    """Top-level configuration for the argunauts alignment pipeline."""

    base_dataset_id: str = DEFAULT_BASE_DATASET_ID
    configs: Sequence[str] = field(default_factory=lambda: DEFAULT_CONFIGS)
    splits: Sequence[str] = field(default_factory=lambda: DEFAULT_SPLITS)
    modes: Sequence[str] = field(default_factory=lambda: DEFAULT_MODES)
    models: Sequence[str] = field(default_factory=lambda: DEFAULT_MODELS)

    max_repair_loops: int = DEFAULT_MAX_REPAIR_LOOPS
    debug: bool = False

    # Max number of (mode, model) groups to process in parallel
    max_group_workers: int = 4

    script_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    assigned_dir: Path = field(init=False)
    aligned_dir: Path = field(init=False)
    merged_dir: Path = field(init=False)
    cleaned_dir: Path = field(init=False)

    mode_config_a: Path = field(init=False)
    mode_config_b: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.script_dir / ("data_debug" if self.debug else "data")
        self.raw_dir = self.data_dir / "raw"
        self.assigned_dir = self.data_dir / "assigned"
        self.aligned_dir = self.data_dir / "aligned"
        self.merged_dir = self.data_dir / "merged"
        self.cleaned_dir = self.data_dir / "cleaned"

        self.mode_config_a = self.script_dir / "argunauts_config_a.yaml"
        self.mode_config_b = self.script_dir / "argunauts_config_b.yaml"


def parse_orchestrator_args(argv: Optional[Sequence[str]] = None) -> OrchestratorConfig:
    """Parse CLI arguments and construct an OrchestratorConfig."""
    parser = argparse.ArgumentParser(
        description="Orchestrate argunauts-thinking alignment with per-sample repair.",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--max-repair-loops",
        type=int,
        default=DEFAULT_MAX_REPAIR_LOOPS,
        help="Maximum repair iterations per (config, split, mode, model) group.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Dataset configs to process.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=DEFAULT_SPLITS,
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=DEFAULT_MODES,
        help="Alignment modes (e.g. a b).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model identifiers to use.",
    )
    parser.add_argument(
        "--base-dataset-id",
        type=str,
        default=DEFAULT_BASE_DATASET_ID,
        help="Base HF dataset ID for raw deep-argmap data.",
    )
    parser.add_argument(
        "--max-group-workers",
        type=int,
        default=4,
        help=(
            "Maximum number of (mode, model) groups to process concurrently per (config, split)."
        ),
    )

    args = parser.parse_args(argv)

    cfg = OrchestratorConfig(
        base_dataset_id=args.base_dataset_id,
        configs=args.configs,
        splits=args.splits if not args.debug else args.splits[:1],
        modes=args.modes,
        models=args.models,
        max_repair_loops=args.max_repair_loops,
        debug=args.debug,
        max_group_workers=args.max_group_workers,
    )
    return cfg


def safe_model_name(model: str) -> str:
    """Return a filesystem-safe version of a model identifier."""
    return model.replace("/", "-")


def create_directories(cfg: OrchestratorConfig) -> None:
    """Create the directory structure used by the orchestrator."""
    for path in (
        cfg.raw_dir,
        cfg.assigned_dir,
        cfg.aligned_dir,
        cfg.merged_dir,
        cfg.cleaned_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def run_subprocess(cmd: List[str], desc: str) -> None:
    """Run a subprocess command, raising on failure."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
        raise RuntimeError(f"Command failed ({desc}): {cmd}") from exc


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def get_mode_config_path(cfg: OrchestratorConfig, mode: str) -> Path:
    return cfg.mode_config_a if mode == "a" else cfg.mode_config_b


def choose_sample_size(cfg: OrchestratorConfig, split: str) -> int:
    if cfg.debug:
        return 5
    return 10000 if split == "train" else 600


def ensure_raw_subset(
    cfg: OrchestratorConfig,
    config_name: str,
    split: str,
    raw_path: Path,
) -> None:
    """Create the raw subset JSON if it does not exist yet."""
    if raw_path.exists():
        return

    n = choose_sample_size(cfg, split)

    cmd: List[str] = [
        "python",
        str(cfg.script_dir / "prepare_subset.py"),
        "--dataset",
        cfg.base_dataset_id,
        "--config-name",
        config_name,
        "--split",
        split,
        "--n",
        str(n),
        "--output",
        str(raw_path),
    ]
    run_subprocess(cmd, desc=f"prepare_subset for {config_name}/{split}")


def ensure_assigned_groups(
    cfg: OrchestratorConfig,
    config_name: str,
    split: str,
) -> None:
    """Run assign_modes_and_models.py if no assigned files exist for this split."""
    raw_path = cfg.raw_dir / f"{config_name}_{split}_raw.json"
    if not raw_path.exists():
        return

    pattern = f"{config_name}_{split}_mode-*_model-*.json"
    if list(cfg.assigned_dir.glob(pattern)):
        return

    cmd: List[str] = [
        "python",
        str(cfg.script_dir / "assign_modes_and_models.py"),
        "--input",
        str(raw_path),
        "--modes",
        *cfg.modes,
        "--models",
        *cfg.models,
        "--output-dir",
        str(cfg.assigned_dir),
        "--seed",
        "42",
    ]
    run_subprocess(cmd, desc=f"assign_modes_and_models for {config_name}/{split}")


def compute_pending_ids(
    original_records: List[Dict[str, Any]],
    canonical_records: List[Dict[str, Any]],
) -> List[Any]:
    """Return example_ids present in original but not yet in canonical."""
    orig_ids = [rec.get("example_id") for rec in original_records]
    canon_ids = {rec.get("example_id") for rec in canonical_records if "example_id" in rec}
    return [ex_id for ex_id in orig_ids if ex_id is not None and ex_id not in canon_ids]


def build_repair_records(
    original_records: List[Dict[str, Any]],
    pending_ids: Iterable[Any],
) -> List[Dict[str, Any]]:
    pending_set = set(pending_ids)
    return [rec for rec in original_records if rec.get("example_id") in pending_set]


def merge_clean_repair_into_canonical(
    canonical_records: List[Dict[str, Any]],
    repair_clean_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id: Dict[Any, Dict[str, Any]] = {
        rec["example_id"]: rec for rec in canonical_records if "example_id" in rec
    }
    for rec in repair_clean_records:
        ex_id = rec.get("example_id")
        if ex_id is None:
            continue
        by_id[ex_id] = rec
    return list(by_id.values())


def run_sdkit_create_cli(
    mode_config: Path,
    input_json: Path,
    model: str,
    output_dir: Path,
    debug: bool = False,
) -> None:
    """Invoke synthetic-data-kit create via CLI."""
    cmd: List[str] = [
        "synthetic-data-kit",
        "-c",
        str(mode_config),
        "create",
        str(input_json),
        "--type",
        "cot-enhance",
        "--model",
        model,
        "--output-dir",
        str(output_dir),
    ]
    if debug:
        cmd.append("--verbose")
    run_subprocess(cmd, desc=f"sdkit create {input_json.name}")


def run_validate_structures_clean_cli(
    original_path: Path,
    transformed_path: Path,
    cleaned_output_path: Path,
    strict_ids: bool,
) -> None:
    """Invoke validate_structures.py in clean mode via CLI."""
    cmd: List[str] = [
        "python",
        str((Path(__file__).resolve().parent / "validate_structures.py")),
        "--original",
        str(original_path),
        "--transformed",
        str(transformed_path),
        "--clean",
        "--clean-output",
        str(cleaned_output_path),
    ]
    if strict_ids:
        cmd.append("--strict-ids")
    run_subprocess(cmd, desc=f"validate_structures clean {transformed_path.name}")


def run_group_alignment_with_repairs_for_group(
    cfg: OrchestratorConfig,
    config_name: str,
    split: str,
    mode: str,
    model: str,
    batch_size: int = 64,
) -> None:
    """Align a single (config, split, mode, model) group with bounded repairs.

    Pending examples are processed in batches of size batch_size within each
    repair loop iteration to reduce per-call load on the SD-Kit CLI while
    preserving the overall retry semantics.
    """
    safe_model = safe_model_name(model)
    assigned_path = cfg.assigned_dir / f"{config_name}_{split}_mode-{mode}_model-{safe_model}.json"
    if not assigned_path.exists():
        return

    mode_config = get_mode_config_path(cfg, mode)
    out_dir = cfg.aligned_dir / config_name / split / f"mode-{mode}_model-{safe_model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{config_name}_{split}_mode-{mode}_model-{safe_model}"
    canonical_path = out_dir / f"{base_name}_enhanced.json"

    for _ in range(cfg.max_repair_loops):
        original_records = load_json(assigned_path)
        canonical_records: List[Dict[str, Any]] = []
        if canonical_path.exists():
            canonical_records = load_json(canonical_path)

        pending_ids = compute_pending_ids(original_records, canonical_records)
        if not pending_ids:
            break

        # Process the pending IDs in fixed-size batches. We compute pending_ids
        # once per outer repair iteration so that structurally invalid examples
        # are only retried in subsequent iterations, matching the original
        # semantics.
        for start in range(0, len(pending_ids), batch_size):
            batch_ids = pending_ids[start : start + batch_size]
            print(
                f"🔁 Processing batch with items {start}-{start + batch_size} of {len(pending_ids)}."
            )

            repair_input_path = out_dir / f"~{base_name}_repair_input.json"
            # SD-Kit follows the convention <input_basename>_enhanced.json for outputs.
            repair_raw_path = out_dir / f"~{base_name}_repair_input_enhanced.json"
            repair_clean_path = out_dir / f"~{base_name}_repair_clean.json"

            repair_records = build_repair_records(original_records, batch_ids)
            save_json(repair_input_path, repair_records)

            # Run SD-Kit on the pending subset; it will create
            # <base_name>_repair_input_enhanced.json in out_dir.
            run_sdkit_create_cli(
                mode_config=mode_config,
                input_json=repair_input_path,
                model=model,
                output_dir=out_dir,
                debug=cfg.debug,
            )

            if not repair_raw_path.exists():
                print(
                    f"⚠️ [WARN] SD-Kit did not write enhanced output "
                    f"{repair_raw_path} for {repair_input_path}. "
                    f"Skipping this batch in this repair loop."
                )
                # Nothing merged; these example_ids remain pending and may be retried
                # in a later outer repair iteration.
                continue

            run_validate_structures_clean_cli(
                original_path=repair_input_path,
                transformed_path=repair_raw_path,
                cleaned_output_path=repair_clean_path,
                strict_ids=True,
            )

            repair_clean_records = load_json(repair_clean_path)
            canonical_records = merge_clean_repair_into_canonical(
                canonical_records,
                repair_clean_records,
            )
            save_json(canonical_path, canonical_records)

            # Best-effort cleanup of per-batch intermediates
            for tmp_path in (repair_input_path, repair_raw_path, repair_clean_path):
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass


def run_group_alignment_with_repairs(
    cfg: OrchestratorConfig,
    config_name: str,
    split: str,
) -> None:
    """Run per-sample repair alignment for all (mode, model) groups in a split.

    Groups (config, split, mode, model) are processed concurrently using a
    thread pool, up to cfg.max_group_workers.
    """
    tasks = []

    total_groups = len(cfg.modes) * len(cfg.models)
    max_workers = min(total_groups, max(1, cfg.max_group_workers))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for mode in cfg.modes:
            for model in cfg.models:
                future = executor.submit(
                    run_group_alignment_with_repairs_for_group,
                    cfg,
                    config_name,
                    split,
                    mode,
                    model,
                )
                tasks.append(((mode, model), future))

        for (mode, model), future in tasks:
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - passthrough
                raise RuntimeError(
                    "Group alignment failed for "
                    f"config={config_name}, split={split}, mode={mode}, model={model}"
                ) from exc


def run_merge_for_config_split(
    cfg: OrchestratorConfig,
    config_name: str,
    split: str,
    merged_output_path: Path,
) -> None:
    """Merge all aligned group outputs into a single split file."""
    if merged_output_path.exists():
        return

    cmd: List[str] = [
        "python",
        str(cfg.script_dir / "merge_per_config.py"),
        "--config",
        config_name,
        "--split",
        split,
        "--input-root",
        str(cfg.aligned_dir),
        "--output",
        str(merged_output_path),
    ]
    run_subprocess(cmd, desc=f"merge_per_config for {config_name}/{split}")


def run_global_validation_and_clean(
    cfg: OrchestratorConfig,
    raw_path: Path,
    merged_path: Path,
    cleaned_output_path: Path,
) -> None:
    """Validate merged output against raw subset and write cleaned version."""
    if not raw_path.exists() or not merged_path.exists():
        return

    run_validate_structures_clean_cli(
        original_path=raw_path,
        transformed_path=merged_path,
        cleaned_output_path=cleaned_output_path,
        strict_ids=True,
    )


def run_config_split_pipeline(cfg: OrchestratorConfig, config_name: str, split: str) -> None:
    """Run the full pipeline for a single (config, split)."""
    raw_path = cfg.raw_dir / f"{config_name}_{split}_raw.json"

    ensure_raw_subset(cfg, config_name, split, raw_path)
    ensure_assigned_groups(cfg, config_name, split)

    run_group_alignment_with_repairs(cfg, config_name, split)

    merged_path = cfg.merged_dir / f"{config_name}-aligned_{split}.json"
    run_merge_for_config_split(cfg, config_name, split, merged_path)

    cleaned_path = cfg.cleaned_dir / f"{config_name}-aligned_{split}.json"
    run_global_validation_and_clean(cfg, raw_path, merged_path, cleaned_path)


def load_local_env(script_dir: Path) -> None:
    """Load a local .env file into the process environment, if present.

    We mirror the behavior of run_alignment.sh, which sourced .env in this
    directory so that API keys and HF tokens are available to SD-Kit and
    publishing scripts.
    """
    env_path = script_dir / ".env"
    if not env_path.exists():
        return

    # Use python-dotenv to load environment variables; if the package is not
    # installed, this import will fail at startup, which is acceptable since
    # the dependency is declared in pyproject.
    load_dotenv(dotenv_path=str(env_path))


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_orchestrator_args(argv)
    # Ensure environment variables from a local .env file are available,
    # matching the behavior of the previous shell orchestrator.
    load_local_env(cfg.script_dir)
    create_directories(cfg)

    for config_name in cfg.configs:
        for split in cfg.splits:
            run_config_split_pipeline(cfg, config_name, split)


if __name__ == "__main__":  # pragma: no cover
    main()
