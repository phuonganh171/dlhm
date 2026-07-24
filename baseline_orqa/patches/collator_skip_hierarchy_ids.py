"""
Patch ORQA collator to skip take/timepoint parsing for hierarchy sample IDs.

Upstream expects ORQA-style ids (``MMOR_..._<int>``) and does
``int(id.rsplit('_', 1)[1])`` for past-visual lookup. Hierarchy ids look like
``001_PKA/anest/.../000000`` and crash. Image-only training leaves
``past_visual_embeds`` empty anyway, so skipping is safe.
Applied by setup.sh after clone.
"""

from __future__ import annotations

from pathlib import Path

OLD = '''            id = feature.pop("id", None)
            if id is not None:
                take_name, take_timepoint = id.rsplit('_', 1)
                take_timepoint = int(take_timepoint)
'''

NEW = '''            id = feature.pop("id", None)
            # Hierarchy baseline ids (take/role/seg/frame) are not ORQA take_timepoint ids.
            if id is not None and ("/" in str(id) or not str(id).rsplit("_", 1)[-1].isdigit()):
                id = None
            if id is not None:
                take_name, take_timepoint = id.rsplit('_', 1)
                take_timepoint = int(take_timepoint)
'''


def patch_collator(orqa_root: Path) -> None:
    path = (
        orqa_root
        / "Qwen2-VL"
        / "LLaMA-Factory"
        / "src"
        / "llamafactory"
        / "data"
        / "collator.py"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text()
    if 'Hierarchy baseline ids' in text:
        print(f"  {path.name} already patched for hierarchy ids")
        return
    if OLD not in text:
        raise RuntimeError(f"Could not patch collator.py — unexpected content in {path}")
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"  patched {path.relative_to(orqa_root)} (skip hierarchy sample ids)")


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <ORQA_ROOT>")
    patch_collator(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
