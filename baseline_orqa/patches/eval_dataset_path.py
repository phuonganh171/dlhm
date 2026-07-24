"""
Patch ORQA LLaMA-Factory to support a separate validation JSON.

Upstream always sets eval_dataset=['orqa'] and resolves the empty file_name
to ``data_json_file``, so train and eval would be the same file. We add
``eval_data_json_file`` / ``eval_cache_file_name`` and wire them in the loader.
"""

from __future__ import annotations

from pathlib import Path


def patch_data_args(orqa_root: Path) -> None:
    path = (
        orqa_root
        / "Qwen2-VL"
        / "LLaMA-Factory"
        / "src"
        / "llamafactory"
        / "hparams"
        / "data_args.py"
    )
    text = path.read_text()
    if "eval_data_json_file" in text:
        print(f"  {path.name} already has eval_data_json_file")
        return

    needle = '''    data_json_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the JSON file containing the dataset."},
    )
'''
    insert = '''    data_json_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the JSON file containing the dataset."},
    )
    eval_data_json_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the JSON file containing the eval/validation dataset."},
    )
    eval_cache_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "Cache path for the tokenized eval dataset (defaults to cache_file_name + '_eval')."},
    )
'''
    if needle not in text:
        raise RuntimeError(f"Could not patch data_args.py — unexpected content in {path}")
    path.write_text(text.replace(needle, insert, 1))
    print(f"  patched {path.relative_to(orqa_root)}")


def patch_loader(orqa_root: Path) -> None:
    path = (
        orqa_root
        / "Qwen2-VL"
        / "LLaMA-Factory"
        / "src"
        / "llamafactory"
        / "data"
        / "loader.py"
    )
    text = path.read_text()

    old_fallback = '''            _cache = getattr(data_args, "eval_cache_file_name", None) or (
                (data_args.cache_file_name + "_eval") if data_args.cache_file_name else None
            )'''
    new_fallback = '''            _cache = getattr(data_args, "eval_cache_file_name", None)
            if _cache is None and data_args.cache_file_name:
                # Keep a file extension (HF datasets rindex(".")) for num_proc>1
                _base = data_args.cache_file_name
                if _base.endswith(".arrow"):
                    _cache = _base[:-6] + "_eval.arrow"
                else:
                    _cache = _base + "_eval.arrow"'''

    # Upgrade already-patched installs that omit the .arrow extension in the fallback
    if old_fallback in text:
        text = text.replace(old_fallback, new_fallback, 1)
        path.write_text(text)
        print(f"  upgraded eval-cache fallback in {path.relative_to(orqa_root)}")
        return

    if "eval_data_json_file" in text and "Hierarchy baseline eval path" in text:
        print(f"  {path.name} already patched for eval_data_json_file")
        return

    old_merge = '''    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args, stage)
'''
    new_merge = '''    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        # Hierarchy baseline eval path: use eval_data_json_file when set (else same as train).
        _train_json = data_args.data_json_file
        _eval_json = getattr(data_args, "eval_data_json_file", None)
        if _eval_json:
            data_args.data_json_file = _eval_json
        try:
            eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args, stage)
        finally:
            data_args.data_json_file = _train_json
'''
    if old_merge not in text:
        raise RuntimeError(f"Could not patch loader.py merge block — unexpected content in {path}")
    text = text.replace(old_merge, new_merge, 1)

    old_cache = '''    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
            cache_file_name=data_args.cache_file_name,
        )
'''
    new_cache = f'''    if not data_args.streaming:
        _cache = data_args.cache_file_name
        if is_eval:
{new_fallback}
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
            cache_file_name=_cache,
        )
'''
    if old_cache not in text:
        raise RuntimeError(f"Could not patch loader.py cache block — unexpected content in {path}")
    path.write_text(text.replace(old_cache, new_cache, 1))
    print(f"  patched {path.relative_to(orqa_root)}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("orqa_root", type=Path)
    args = parser.parse_args()
    patch_data_args(args.orqa_root)
    patch_loader(args.orqa_root)
    print("  eval dataset path patches applied")


if __name__ == "__main__":
    main()
