"""
Patch ORQA's Qwen2-VL for image-only hierarchy training.

Makes PointTransformer / audio optional so we do not need spconv for the
hierarchy baseline (RGB multi-view only). Applied by setup.sh after clone.
"""

from __future__ import annotations

from pathlib import Path


STUB_POINTTRANSFORMER = '''"""Stub PointTransformer for image-only hierarchy baseline (no spconv)."""

import torch
import torch.nn as nn


class Point(dict):
    """Minimal stand-in for Pointcept Point."""

    pass


class PointTransformerV3(nn.Module):
    """No-op PC encoder — hierarchy baseline never passes point clouds."""

    def __init__(self, cls_mode=True, project_pc_dim=1536):
        super().__init__()
        self.project_pc = nn.Linear(6, project_pc_dim)

    def _init_weights(self, dtype=None, device=None):
        return

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Point clouds are disabled in the hierarchy ORQA baseline"
        )
'''


def patch_pointtransformerv3(orqa_root: Path) -> None:
    target = (
        orqa_root
        / "Qwen2-VL"
        / "LLaMA-Factory"
        / "src"
        / "llamafactory"
        / "model"
        / "qwen2_vl"
        / "pointtransformerv3.py"
    )
    if not target.exists():
        raise FileNotFoundError(target)
    # Keep a backup once
    backup = target.with_suffix(".py.orqa_orig")
    if not backup.exists():
        backup.write_text(target.read_text())
    target.write_text(STUB_POINTTRANSFORMER)
    print(f"  patched {target.relative_to(orqa_root)} (image-only stub)")


def patch_workflow_point_calls(orqa_root: Path) -> None:
    """Guard point_transformer init/float so missing attrs are OK."""
    workflow = (
        orqa_root
        / "Qwen2-VL"
        / "LLaMA-Factory"
        / "src"
        / "llamafactory"
        / "train"
        / "sft"
        / "workflow.py"
    )
    text = workflow.read_text()
    if "getattr(model.visual.image_pooler, 'point_transformer'" in text:
        print("  workflow.py already patched")
        return

    old_init = (
        "        model.visual.image_pooler.point_transformer._init_weights("
        "dtype=torch.float32, device=training_args.device)  # important!"
    )
    new_init = (
        "        _pt = getattr(model.visual.image_pooler, 'point_transformer', None)\n"
        "        if _pt is not None and hasattr(_pt, '_init_weights'):\n"
        "            _pt._init_weights(dtype=torch.float32, device=training_args.device)"
    )
    old_float = "    model.visual.image_pooler.point_transformer.float()"
    new_float = (
        "    _pt = getattr(model.visual.image_pooler, 'point_transformer', None)\n"
        "    if _pt is not None:\n"
        "        _pt.float()"
    )
    if old_init not in text or old_float not in text:
        print("  WARN: workflow.py unexpected; check manually")
        return
    text = text.replace(old_init, new_init).replace(old_float, new_float)
    workflow.write_text(text)
    print(f"  patched {workflow.relative_to(orqa_root)}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("orqa_root", type=Path)
    args = parser.parse_args()
    patch_pointtransformerv3(args.orqa_root)
    patch_workflow_point_calls(args.orqa_root)
    print("  image-only patches applied")


if __name__ == "__main__":
    main()
