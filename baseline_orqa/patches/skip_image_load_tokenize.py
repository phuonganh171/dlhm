"""
Skip loading images during tokenization when ``fix_number_of_image_tokens`` is set.

ORQA's Qwen2VLPlugin.process_messages always called ``_get_mm_inputs`` (opens every
image) even when a fixed token count is configured and images are ignored for
token placement. With ~100k hierarchy samples on a rclone FUSE mount, that
hammering causes ``OSError: [Errno 5] Input/output error`` mid-preprocess.

Also retries Image.open on transient I/O errors (still needed at train time).
Applied by setup.sh after clone.
"""

from __future__ import annotations

from pathlib import Path

SKIP_OLD = '''        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.fix_number_of_image_tokens is not None:
'''

SKIP_NEW = '''        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        # Hierarchy / fixed-token mode: do not open images during tokenize
        # (NAS FUSE EIO under num_proc>1). Collator still loads at train time.
        if self.fix_number_of_image_tokens is not None:
            image_grid_thw, video_grid_thw = [], []
        else:
            mm_inputs = self._get_mm_inputs(images, videos, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.fix_number_of_image_tokens is not None:
'''

OPEN_OLD = '''            if isinstance(image, str):
                if 'simstation/camera01' in image:
                    # add highres_needed to flag to kwargs
                    kwargs['highres_needed'] = True
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])
'''

OPEN_NEW = '''            if isinstance(image, str):
                if 'simstation/camera01' in image:
                    # add highres_needed to flag to kwargs
                    kwargs['highres_needed'] = True
                image = _open_image_with_retry(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = _open_image_with_retry(image["path"])
'''

HELPER = '''
def _open_image_with_retry(path, retries: int = 5, base_sleep: float = 0.5):
    """Open image path with retries for transient FUSE / NAS I/O errors."""
    import time

    last_err = None
    for attempt in range(retries):
        try:
            return Image.open(path)
        except OSError as err:
            last_err = err
            # Errno 5 (EIO), 116 (ESTALE), etc. — brief backoff then retry
            if attempt + 1 >= retries:
                break
            time.sleep(base_sleep * (2 ** attempt))
    raise last_err

'''


def patch_mm_plugin(orqa_root: Path) -> None:
    path = (
        orqa_root
        / "Qwen2-VL"
        / "LLaMA-Factory"
        / "src"
        / "llamafactory"
        / "data"
        / "mm_plugin.py"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text()
    changed = False

    if "Hierarchy / fixed-token mode: do not open images during tokenize" in text:
        print(f"  {path.name} already patched for skip-image tokenize")
    else:
        if SKIP_OLD not in text:
            raise RuntimeError(f"Could not patch process_messages in {path}")
        text = text.replace(SKIP_OLD, SKIP_NEW, 1)
        changed = True
        print(f"  patched {path.relative_to(orqa_root)} (skip image load when fixed tokens)")

    if "_open_image_with_retry" in text:
        print(f"  {path.name} already patched for image open retry")
    else:
        if OPEN_OLD not in text:
            raise RuntimeError(f"Could not patch _regularize_images in {path}")
        # Insert helper after imports / before first class-ish content
        marker = "ImageInput = Union[str, bytes, EncodedImage, ImageObject]\n    VideoInput = str\n"
        # Prefer inserting just before BasePlugin class
        class_marker = "\nclass BasePlugin:"
        if class_marker not in text:
            raise RuntimeError(f"Could not find BasePlugin in {path}")
        text = text.replace(class_marker, HELPER + class_marker, 1)
        text = text.replace(OPEN_OLD, OPEN_NEW, 1)
        changed = True
        print(f"  patched {path.relative_to(orqa_root)} (retry Image.open on I/O errors)")

    if changed:
        path.write_text(text)


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <ORQA_ROOT>")
    patch_mm_plugin(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
