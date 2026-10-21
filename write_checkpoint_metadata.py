#!/usr/bin/env python3
"""
write_checkpoint_metadata.py

Automated metadata generator for TerraCIL / SatReCIL PyTorch checkpoints (.pt)
and experience replay buffers (.json).

Reads trained PyTorch checkpoint weights, extracts model topology, class mappings,
parameter counts, size footprints, and test evaluation metrics (Top-1, Top-2, Macro-F1),
then writes standardized metadata JSON files:
    - stage1_metadata.json
    - stage2_metadata.json
    - stage3_metadata.json
    - checkpoint_manifest.json (consolidated registry of all checkpoints)

Usage:
    python write_checkpoint_metadata.py
    python write_checkpoint_metadata.py --checkpoint-dir checkpoints --verbose
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn

# Standard class mapping for EuroSAT staged incremental learning
DEFAULT_STAGE_CLASSES = {
    1: ["Forest", "Residential", "River"],
    2: ["Forest", "Residential", "River", "Highway"],
    3: ["Forest", "Residential", "River", "Highway", "Industrial"],
}

CLASS_EMOJIS = {
    "Forest": "🌲",
    "Residential": "🏘️",
    "River": "🌊",
    "Highway": "🛣️",
    "Industrial": "🏭",
}


def compute_param_stats(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Compute backbone, classifier, and total parameter counts and sizes from state_dict."""
    backbone_params = 0
    classifier_params = 0

    for name, tensor in state_dict.items():
        count = tensor.numel()
        if "classifier" in name or "fc" in name or "head" in name:
            classifier_params += count
        else:
            backbone_params += count

    total_params = backbone_params + classifier_params
    counts = {
        "backbone": backbone_params,
        "classifier": classifier_params,
        "total": total_params,
        "trainable": total_params,
    }
    sizes = {
        "fp32_kb": round((total_params * 4) / 1024.0, 2),
        "int8_kb": round((total_params * 1) / 1024.0, 2),
        "total_params": total_params,
    }
    return counts, sizes


def load_replay_buffer_summary(checkpoint_dir: Path, stage_num: int, seed: int = 42) -> Optional[Dict[str, Any]]:
    """Inspect replay buffer JSON if present to enrich metadata with replay exemplars."""
    replay_filename = f"stage{stage_num}_replay_seed{seed}.json"
    candidates = [
        checkpoint_dir / replay_filename,
        checkpoint_dir.parent / "checkpoints" / replay_filename,
        checkpoint_dir.parent / "SatReCIL" / "checkpoints" / replay_filename,
    ]
    for p in candidates:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                exemplars = data.get("exemplars", {})
                per_class_counts = {int(k): len(v) for k, v in exemplars.items()}
                return {
                    "replay_checkpoint_file": replay_filename,
                    "capacity": data.get("capacity", 500),
                    "seed": data.get("seed", seed),
                    "total_exemplars": sum(per_class_counts.values()),
                    "exemplars_per_class": per_class_counts,
                }
            except Exception:
                pass
    return None


def extract_metadata_from_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Parse a .pt checkpoint file and return structured metadata dictionary."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected dict in checkpoint {checkpoint_path.name}, got {type(ckpt).__name__}")

    # Determine state dict
    state_dict = None
    for k in ("model_state_dict", "state_dict", "model", "net", "weights"):
        if isinstance(ckpt.get(k), dict):
            state_dict = ckpt[k]
            break
    if state_dict is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt

    if state_dict is None:
        raise ValueError(f"Could not locate model state_dict in {checkpoint_path.name}")

    # Determine stage and method from checkpoint or filename
    filename = checkpoint_path.name
    stage = ckpt.get("stage")
    if stage is None:
        if "stage1" in filename:
            stage = 1
        elif "stage2" in filename:
            stage = 2
        elif "stage3" in filename:
            stage = 3
        else:
            stage = 1

    method = ckpt.get("method")
    if method is None:
        if "base" in filename:
            method = "base"
        elif "replay" in filename:
            method = "replay"
        elif "joint" in filename:
            method = "joint"
        elif "naive" in filename:
            method = "naive"
        else:
            method = "replay"

    seed = ckpt.get("seed", 42)

    # Class names and count
    stage_classes = ckpt.get("stage_classes")
    if not stage_classes:
        stage_classes = DEFAULT_STAGE_CLASSES.get(stage, ["Forest", "Residential", "River"])

    num_classes = ckpt.get("num_classes", len(stage_classes))
    class_to_idx = ckpt.get("class_to_idx", {c: i for i, c in enumerate(stage_classes)})

    # Parameters and size
    counts = ckpt.get("param_counts")
    sizes = ckpt.get("size_info")
    if not counts or not sizes:
        counts, sizes = compute_param_stats(state_dict)

    # Test metrics
    test_metrics = ckpt.get("test_metrics", {})
    top1 = float(test_metrics.get("top1_accuracy", 0.0))
    top2 = float(test_metrics.get("top2_accuracy", 0.0))
    macro_f1 = float(test_metrics.get("macro_f1", 0.0))

    # Replay buffer state
    replay_summary = load_replay_buffer_summary(checkpoint_path.parent, stage, seed)

    metadata = {
        "stage": stage,
        "method": method,
        "seed": seed,
        "checkpoint_file": filename,
        "num_classes": num_classes,
        "class_names": stage_classes,
        "class_to_idx": class_to_idx,
        "feature_dim": 64,
        "input_resolution": [3, 64, 64],
        "parameter_counts": counts,
        "size_kb": sizes,
        "top1_accuracy": top1,
        "top2_accuracy": top2,
        "macro_f1": macro_f1,
        "metrics_summary": {
            "top1_accuracy_percent": f"{top1 * 100:.2f}%",
            "top2_accuracy_percent": f"{top2 * 100:.2f}%",
            "macro_f1": round(macro_f1, 4),
        },
        "per_class_metrics": test_metrics.get("per_class", {}),
        "replay_buffer": replay_summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Add continual learning config if stored
    if "continual_learning_config" in ckpt:
        metadata["continual_learning_config"] = ckpt["continual_learning_config"]

    return metadata


def write_metadata_file(metadata: Dict[str, Any], output_path: Path, verbose: bool = True) -> Path:
    """Save metadata dict to formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if verbose:
        print(f"[+] Saved metadata: {output_path.name} (Stage {metadata['stage']} - {metadata['num_classes']} classes)")
    return output_path


def process_checkpoints(
    checkpoint_dir: Path,
    output_dir: Optional[Path] = None,
    sync_satrecil: bool = True,
    verbose: bool = True,
) -> List[Path]:
    """
    Scan checkpoint directory, extract metadata for all .pt checkpoints,
    and generate standard stage metadata files.
    """
    if output_dir is None:
        output_dir = checkpoint_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(list(checkpoint_dir.glob("*.pt")))
    if not pt_files:
        print(f"[WARN] No .pt files found in {checkpoint_dir.resolve()}")
        return []

    if verbose:
        print(f"[INFO] Discovered {len(pt_files)} PyTorch checkpoints in {checkpoint_dir.resolve()}")

    generated_files: List[Path] = []
    manifest: Dict[str, Any] = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_directory": str(checkpoint_dir.resolve()),
        "stages": {},
        "all_checkpoints": [],
    }

    # Primary mapping for canonical stage checkpoints
    primary_stage_targets = {
        1: ("stage1_base_seed42.pt", output_dir / "stage1_metadata.json"),
        2: ("stage2_replay_seed42.pt", output_dir / "stage2_metadata.json"),
        3: ("stage3_replay_seed42.pt", output_dir / "stage3_metadata.json"),
    }

    for pt_path in pt_files:
        try:
            meta = extract_metadata_from_checkpoint(pt_path)
            stage_num = meta["stage"]
            method = meta["method"]

            # 1. Write specific metadata for this checkpoint file
            specific_meta_path = output_dir / f"{pt_path.stem}_metadata.json"
            write_metadata_file(meta, specific_meta_path, verbose=verbose)
            generated_files.append(specific_meta_path)

            # 2. If this is the primary checkpoint for the stage, write canonical stageX_metadata.json
            if stage_num in primary_stage_targets and pt_path.name == primary_stage_targets[stage_num][0]:
                canonical_path = primary_stage_targets[stage_num][1]
                write_metadata_file(meta, canonical_path, verbose=verbose)
                generated_files.append(canonical_path)
                manifest["stages"][f"stage{stage_num}"] = {
                    "checkpoint": pt_path.name,
                    "metadata": canonical_path.name,
                    "num_classes": meta["num_classes"],
                    "classes": meta["class_names"],
                    "top1_accuracy": meta["top1_accuracy"],
                    "macro_f1": meta["macro_f1"],
                }

            manifest["all_checkpoints"].append({
                "filename": pt_path.name,
                "stage": stage_num,
                "method": method,
                "num_classes": meta["num_classes"],
                "classes": meta["class_names"],
                "top1_accuracy": meta["top1_accuracy"],
                "macro_f1": meta["macro_f1"],
                "params": meta["parameter_counts"]["total"],
                "size_fp32_kb": meta["size_kb"]["fp32_kb"],
            })

        except Exception as exc:
            print(f"[ERROR] Processing {pt_path.name}: {exc}", file=sys.stderr)

    # 3. Write consolidated manifest
    manifest_path = output_dir / "checkpoint_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    generated_files.append(manifest_path)
    if verbose:
        print(f"[+] Saved consolidated manifest: {manifest_path.name}")

    # 4. Sync to SatReCIL/checkpoints if requested and exists
    if sync_satrecil:
        satrecil_ckpt = checkpoint_dir.parent / "SatReCIL" / "checkpoints"
        if satrecil_ckpt.is_dir() and satrecil_ckpt.resolve() != output_dir.resolve():
            for src_p in [
                output_dir / "stage1_metadata.json",
                output_dir / "stage2_metadata.json",
                output_dir / "stage3_metadata.json",
                manifest_path,
            ]:
                if src_p.is_file():
                    dest_p = satrecil_ckpt / src_p.name
                    dest_p.write_bytes(src_p.read_bytes())
                    if verbose:
                        print(f"[+] Synchronized to SatReCIL: {dest_p.name}")

    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract and write checkpoint metadata JSON files for TerraCIL / SatReCIL."
    )
    parser.add_argument(
        "--checkpoint-dir",
        "-c",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing PyTorch .pt checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory where JSON metadata should be written (default: same as --checkpoint-dir)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not copy metadata to SatReCIL/checkpoints",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output",
    )

    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir

    if not checkpoint_dir.is_dir():
        # Fallback candidate check
        alt = Path(__file__).resolve().parent / "checkpoints"
        if alt.is_dir():
            checkpoint_dir = alt
        else:
            print(f"[ERROR] Checkpoints directory '{checkpoint_dir}' does not exist.", file=sys.stderr)
            sys.exit(1)

    generated = process_checkpoints(
        checkpoint_dir=checkpoint_dir,
        output_dir=args.output_dir,
        sync_satrecil=not args.no_sync,
        verbose=not args.quiet,
    )

    if generated:
        print(f"\n[DONE] Success: Generated {len(generated)} metadata files.")
    else:
        print("\n[WARN] No metadata files generated.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
