#!/usr/bin/env python3
"""
classify_image.py  --  Standalone CLI classifier for SatReCIL / TerraCIL
=========================================================================
Loads a trained EuroSatCompactCNN checkpoint and classifies an input image.

Usage examples:
    python classify_image.py path/to/image.png
    python classify_image.py path/to/image.png --stage 2
    python classify_image.py path/to/image.png --all-stages
    python classify_image.py path/to/image.png --checkpoint checkpoints/stage3_joint_seed42.pt
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps

# ─── EuroSAT normalisation stats (RGB) ───────────────────────────────
EUROSAT_MEAN = (0.3444, 0.3803, 0.4078)
EUROSAT_STD  = (0.2026, 0.1369, 0.1158)
IMAGE_SIZE   = 64

# ─── Default checkpoint directory (relative to this script) ──────────
SCRIPT_DIR      = Path(__file__).resolve().parent
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"

# Default replay checkpoints per stage
DEFAULT_CKPT = {
    1: "stage1_base_seed42.pt",
    2: "stage2_replay_seed42.pt",
    3: "stage3_replay_seed42.pt",
}


# ─── Model definition (self-contained, matches SatReCIL exactly) ───
class EuroSatCompactCNN(nn.Module):
    """Compact 4-block CNN for EuroSAT 64x64 land-cover classification.

    Architecture (SatReCIL backbone, ~60,512 params):
      conv1: 3  -> 16, 3x3, pad 1, ReLU + MaxPool2d(2)  [32x32]
      conv2: 16 -> 32, 3x3, pad 1, ReLU + MaxPool2d(2)  [16x16]
      conv3: 32 -> 64, 3x3, pad 1, ReLU + MaxPool2d(2)  [8x8]
      conv4: 64 -> 64, 3x3, pad 1, ReLU                  [8x8] (NO maxpool)
      pool:  AdaptiveAvgPool2d((1, 1))                   [64]
      classifier: Linear(64, num_classes)
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))  # Note: conv4 has no maxpool in SatReCIL
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)

    def get_parameter_counts(self):
        backbone = sum(p.numel() for m in [self.conv1, self.conv2, self.conv3, self.conv4] for p in m.parameters())
        classifier = sum(p.numel() for p in self.classifier.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"backbone": backbone, "classifier": classifier, "total": total}

    def get_model_size(self):
        counts = self.get_parameter_counts()
        return {
            "fp32_kb": round((counts["total"] * 4) / 1024.0, 2),
            "int8_kb": round((counts["total"] * 1) / 1024.0, 2),
            "total_params": counts["total"],
        }


# ─── Helpers ─────────────────────────────────────────────────────────

def get_transform():
    """Preprocessing pipeline matching training: resize + normalise."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
    ])


def load_checkpoint(ckpt_path: Path):
    """Load a .pt checkpoint and return (model, class_names, metadata)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    num_classes  = ckpt["num_classes"]
    class_names  = ckpt["stage_classes"]          # ordered list
    class_to_idx = ckpt.get("class_to_idx", {})   # {name: idx}

    model = EuroSatCompactCNN(num_classes=num_classes)

    # Clean state-dict keys (strip 'module.' prefix if present)
    raw_sd = ckpt["model_state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    meta = {
        "stage":        ckpt.get("stage"),
        "method":       ckpt.get("method", "base"),
        "seed":         ckpt.get("seed"),
        "test_metrics": ckpt.get("test_metrics"),
    }
    return model, class_names, meta


def classify(model, image_tensor, class_names):
    """Run inference and return sorted list of (class_name, probability)."""
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))   # (1, C)
        probs  = F.softmax(logits, dim=1).squeeze()  # (C,)

    results = []
    for idx, name in enumerate(class_names):
        results.append((name, probs[idx].item()))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ─── Pretty-printing ────────────────────────────────────────────────

def print_results(results, stage_label: str = "", meta: dict = None):
    """Print a nicely formatted classification result table."""
    predicted = results[0]

    if stage_label:
        print(f"\n{'=' * 50}")
        print(f"  {stage_label}")
        print(f"{'=' * 50}")
    else:
        print(f"\n{'=' * 50}")

    if meta:
        info_parts = []
        if meta.get("stage") is not None:
            info_parts.append(f"Stage {meta['stage']}")
        if meta.get("method"):
            info_parts.append(f"Method: {meta['method']}")
        if info_parts:
            print(f"  [{' | '.join(info_parts)}]")
        tm = meta.get("test_metrics")
        if tm:
            acc = tm.get("top1_accuracy") or tm.get("accuracy")
            if acc is not None:
                # If value is already in percentage form (>1), print directly
                if acc > 1:
                    print(f"  Test accuracy: {acc:.2f}%")
                else:
                    print(f"  Test accuracy: {acc*100:.2f}%")

    print(f"\n  Prediction:  {predicted[0]}  ({predicted[1]*100:.2f}%)")
    print(f"\n  {'Class':<20} {'Confidence':>10}")
    print(f"  {'-'*20} {'-'*10}")
    for name, prob in results:
        bar = '#' * int(prob * 30)
        print(f"  {name:<20} {prob*100:>9.2f}%  {bar}")

    print()


# ─── Main ────────────────────────────────────────────────────────────

def main():
    # Fix Windows console encoding for safety
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Classify a satellite image using SatReCIL checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classify_image.py sample_images/sample_river.png
  python classify_image.py my_image.jpg --stage 2
  python classify_image.py my_image.jpg --all-stages
  python classify_image.py my_image.jpg --checkpoint checkpoints/stage3_joint_seed42.pt
        """,
    )
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3], default=3,
        help="Which stage checkpoint to use (default: 3 = all 5 classes)",
    )
    parser.add_argument(
        "--all-stages", action="store_true",
        help="Run the image through all 3 stages and compare results",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a specific .pt checkpoint (overrides --stage)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Show only the top-K predictions (default: show all)",
    )
    args = parser.parse_args()

    # ── Validate image path ──────────────────────────────────────────
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load and preprocess image ────────────────────────────────────
    try:
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        print(f"ERROR: Could not open image: {e}", file=sys.stderr)
        sys.exit(1)

    transform = get_transform()
    img_tensor = transform(img)

    w, h = img.size
    print(f"\nImage: {img_path.name}  ({w}x{h} px)")

    # ── Determine which checkpoint(s) to run ─────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        stages_to_run = [(ckpt_path, None)]
    elif args.all_stages:
        stages_to_run = []
        for s in [1, 2, 3]:
            p = CHECKPOINTS_DIR / DEFAULT_CKPT[s]
            if p.exists():
                stages_to_run.append((p, f"Stage {s}"))
            else:
                print(f"WARNING: Stage {s} checkpoint not found: {p}")
        if not stages_to_run:
            print("ERROR: No checkpoints found!", file=sys.stderr)
            sys.exit(1)
    else:
        s = args.stage
        p = CHECKPOINTS_DIR / DEFAULT_CKPT[s]
        if not p.exists():
            print(f"ERROR: Checkpoint not found: {p}", file=sys.stderr)
            sys.exit(1)
        stages_to_run = [(p, f"Stage {s}")]

    # ── Run inference ────────────────────────────────────────────────
    for ckpt_path, label in stages_to_run:
        try:
            model, class_names, meta = load_checkpoint(ckpt_path)
        except Exception as e:
            print(f"ERROR loading {ckpt_path.name}: {e}", file=sys.stderr)
            continue

        results = classify(model, img_tensor, class_names)

        if args.top_k:
            results = results[:args.top_k]

        lbl = label or ckpt_path.stem
        print_results(results, stage_label=lbl, meta=meta)

    print("Done.")


if __name__ == "__main__":
    main()
