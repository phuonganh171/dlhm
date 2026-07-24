"""
Patch ORQA aligner so missing ``pc`` / ``audio`` / ``segmasks`` columns are OK.

Image-only hierarchy samples omit point clouds and audio; upstream still maps
those columns in dataset_info and indexes them with ``example[key]`` (KeyError).
Applied by setup.sh after clone.
"""

from __future__ import annotations

from pathlib import Path

OLD = '''        "_pc": example[dataset_attr.pc] if dataset_attr.pc else "",
        "_segmasks": example[dataset_attr.segmasks] if dataset_attr.segmasks else "",
        "_audio": example[dataset_attr.audio] if dataset_attr.audio else "",
        "_id": example[dataset_attr.id] if dataset_attr.id else ""
'''

NEW = '''        # .get: image-only samples may omit optional multimodal columns
        "_pc": example.get(dataset_attr.pc, "") if dataset_attr.pc else "",
        "_segmasks": example.get(dataset_attr.segmasks, "") if dataset_attr.segmasks else "",
        "_audio": example.get(dataset_attr.audio, "") if dataset_attr.audio else "",
        "_id": example.get(dataset_attr.id, "") if dataset_attr.id else ""
'''


def patch_aligner(orqa_root: Path) -> None:
    path = (
        orqa_root
        / "Qwen2-VL"
        / "LLaMA-Factory"
        / "src"
        / "llamafactory"
        / "data"
        / "aligner.py"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text()
    if "example.get(dataset_attr.pc" in text:
        print(f"  {path.name} already patched for optional pc/audio")
        return
    if OLD not in text:
        raise RuntimeError(f"Could not patch aligner.py — unexpected content in {path}")
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"  patched {path.relative_to(orqa_root)} (optional pc/audio)")


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <ORQA_ROOT>")
    patch_aligner(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
