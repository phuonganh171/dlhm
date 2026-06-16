"""
Shared utilities for MM-OR scene graph processing.

Provides:
  - Entity / predicate vocabulary and human-readable label mapping
  - Data loading (frame map, relation labels, timestamps)
  - LLM model loading (4-bit NF4 quantization)
  - Inference helpers (chat template, batch generation, think-tag stripping)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera key mapping
# ---------------------------------------------------------------------------

COLORIMAGE_CAMERA_KEYS = {
    "azure": "camera01",
    "simstation": "camera02",
    "trackercam": "camera03",
    "tracker": "camera04",
}

# ---------------------------------------------------------------------------
# Entity / predicate vocabulary
# ---------------------------------------------------------------------------

ENTITY_NAMES = {
    "patient":          "patient",
    "nurse":            "scrub nurse",
    "mps":              "robot technician (MPS)",
    "instrument_table": "instrument table",
    "anest":            "anaesthetist",
    "ot":               "operating table",
    "drape":            "surgical drape",
    "circulator":       "circulator nurse",
    "head_surgeon":     "head surgeon",
    "ae":               "anaesthesia equipment",
    "assistant_surgeon":"assistant surgeon",
    "instrument":       "surgical instrument",
    "mps_station":      "MPS workstation",
    "mako_robot":       "Mako robotic arm",
    "monitor":          "monitor",
    "tracker":          "infrared tracker",
    "saw":              "bone saw",
    "drill":            "drill",
    "c_arm":            "C-arm",
    "hammer":           "hammer",
    "unrelated_person": "unrelated person",
    "robot_setup":      "robot setup (physical)",
    "robot_monitor":    "robot monitor (Mako screen)",
}

PREDICATE_NAMES = {
    "CloseTo":     "is close to",
    "LyingOn":     "is lying on",
    "Holding":     "is holding",
    "Manipulating":"is manipulating",
    "Preparing":   "is preparing",
    "Assisting":   "is assisting",
    "Touching":    "is touching",
    "Calibrating": "is calibrating",
    "Drilling":    "is drilling",
    "Sawing":      "is sawing",
    "Hammering":   "is hammering",
    "Suturing":    "is suturing",
    "Cutting":     "is cutting",
    "Cementing":   "is cementing",
    "Cleaning":    "is cleaning",
    "Scanning":    "is scanning",
}

PASSIVE_PREDICATES = {"CloseTo", "LyingOn"}
ACTIVE_PREDICATES = set(PREDICATE_NAMES.keys()) - PASSIVE_PREDICATES

# Roles = entities that appear as subjects (people / agents)
ROLE_ENTITIES = {
    "head_surgeon", "assistant_surgeon", "nurse", "circulator",
    "anest", "mps", "unrelated_person",
}

# Tools / instruments = entities that appear as the *object* of handling/action
# relations (Holding, Touching, Manipulating, ...).  Treated as roles via an
# inverse extraction (group by object instead of subject).
TOOL_ENTITIES = {
    "saw", "drill", "hammer", "c_arm", "tracker", "instrument",
}

# Action verbs whose triplet targets the patient (not the tool) but which imply
# a specific tool being in use.  Used to attribute the action to its tool's
# timeline, so e.g. (head_surgeon, Sawing, patient) marks the saw as in use.
VERB_TOOL = {
    "Sawing": "saw",
    "Drilling": "drill",
    "Hammering": "hammer",
    "Scanning": "c_arm",
}


def humanize(entity: str) -> str:
    """Map dataset entity id to a stable human-readable name."""
    return ENTITY_NAMES.get(entity, entity.replace("_", " "))


def humanize_pred(pred: str) -> str:
    """Map dataset predicate id to a stable human-readable phrase."""
    return PREDICATE_NAMES.get(pred, pred)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_frame_map(take_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Index-based mapping from relation-label timepoint_id -> frame ids.

    Relation-label files are numbered sequentially (000000 ... 001798) and
    correspond 1-to-1 with the *array index* of each entry in
    timestamp_to_pcd_and_frames_list.json.  The JSON key (entry[0]) is the
    tracker / pcd frame number, which may have gaps due to dropped frames
    and therefore does NOT match the relation-label numbering after index 443.
    """
    fpath = take_dir / "timestamp_to_pcd_and_frames_list.json"
    data = json.loads(fpath.read_bytes().decode("utf-8"))
    return {f"{i:06d}": entry[1] for i, entry in enumerate(data)}


def load_relation_labels(
    take_dir: Path, frame_map: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, list]]:
    """
    Returns a sorted list of (timepoint_id, triplets) for all relation_label files.
    If timepoint_id is not in frame_map, scene graph is [] (no mapping to colorimage).
    """
    labels_dir = take_dir / "relation_labels"
    entries = []
    unmapped = 0
    for fpath in sorted(labels_dir.glob("*.json")):
        try:
            timepoint_id = fpath.stem
            if timepoint_id not in frame_map:
                entries.append((timepoint_id, []))
                unmapped += 1
                continue
            data = json.loads(fpath.read_bytes().decode("utf-8"))
            entries.append((timepoint_id, data["rel_annotations"]))
        except Exception:
            continue
    if unmapped:
        logger.info(
            "%d timepoints have no entry in timestamp_to_pcd_and_frames_list.json "
            "(empty scene graph).",
            unmapped,
        )
    return entries


def load_robot_phase(take_dir: Path) -> Dict[str, str]:
    """
    Map timepoint_id -> physical robot setup phase string.

    The file (take_timestamp_to_robot_phase/<take>.json) is a sibling of
    take_dir's parent and is keyed by the *same* timepoint ids used by the
    relation labels / frame map, so no frame remapping is needed.
    Returns {} if the file is absent.
    """
    fpath = take_dir.parent / "take_timestamp_to_robot_phase" / f"{take_dir.name}.json"
    if not fpath.exists():
        logger.info("No robot-phase file at %s", fpath)
        return {}
    return json.loads(fpath.read_bytes().decode("utf-8"))


def load_screen_summaries(
    take_dir: Path, frame_map: Dict[str, Dict[str, Any]]
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Map timepoint_id -> (phase, current_step) from the Mako navigation screen.

    Screen-summary files (screen_summaries/<take>/<frame>.json) are keyed by the
    *simstation* frame number, so each timepoint is routed through
    frame_map[tp]['simstation'].  When a frame shows a transition flicker with
    multiple phases / steps, the highest-certainty entry of each type is kept.
    Timepoints with no mapping or no file are simply omitted.
    """
    ss_dir = take_dir.parent / "screen_summaries" / take_dir.name
    result: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    if not ss_dir.exists():
        logger.info("No screen-summaries dir at %s", ss_dir)
        return result

    for tp_id, raw in frame_map.items():
        sim = raw.get("simstation")
        if sim is None:
            continue
        fpath = ss_dir / f"{sim}.json"
        if not fpath.exists():
            continue
        try:
            data = json.loads(fpath.read_bytes().decode("utf-8"))
        except Exception:
            continue

        phase: Optional[str] = None
        step: Optional[str] = None
        best_phase_cert = -1.0
        best_step_cert = -1.0
        for name, info in data.items():
            kind = info.get("type")
            try:
                cert = float(info.get("certainty", 0))
            except (TypeError, ValueError):
                cert = 0.0
            if kind == "phase" and cert > best_phase_cert:
                phase, best_phase_cert = name, cert
            elif kind == "current_step" and cert > best_step_cert:
                step, best_step_cert = name, cert

        if phase is not None or step is not None:
            result[tp_id] = (phase, step)
    return result


def frame_info(frame_map: Dict[str, Dict[str, Any]], tp_id: str) -> Optional[Dict[str, Any]]:
    """Per-second frame record for JSONL, or None if unmapped."""
    raw = frame_map.get(tp_id)
    if raw is None:
        return None
    info: Dict[str, Any] = {"original_timestamp": raw["original_timestamp"]}
    for src_key, cam_name in COLORIMAGE_CAMERA_KEYS.items():
        if src_key in raw:
            info[cam_name] = raw[src_key]
    return info


def original_timestamp(frame_map: Dict[str, Dict[str, Any]], tp_id: str) -> Optional[int]:
    raw = frame_map.get(tp_id)
    return raw["original_timestamp"] if raw else None


def make_windows(entries: list, window_size: int) -> List[list]:
    """Split timepoint list into non-overlapping windows of `window_size`."""
    return [
        entries[i : i + window_size]
        for i in range(0, len(entries), window_size)
    ]


# ---------------------------------------------------------------------------
# LLM model loading & inference
# ---------------------------------------------------------------------------

def load_model(model_name_or_path: str, hf_token: Optional[str]):
    """Load model + tokenizer.  Uses 4-bit NF4 quantization by default so
    that large models (e.g. Qwen3-32B) fit on a single 24 GB GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading tokenizer from: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading model from: %s (4-bit NF4)", model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks from generated text."""
    return _THINK_RE.sub("", text).strip()


def build_chat_input(tokenizer, system: str, user: str, *, enable_thinking: bool = True) -> str:
    """Apply the model's chat template (works with Qwen3, Qwen2.5, Llama, etc.)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if not enable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _get_model_device(model):
    """Return the device of the first parameter (safe for quantized / device_map models)."""
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
) -> List[str]:
    """Tokenize, generate, decode, and strip <think> blocks."""
    import torch

    device = _get_model_device(model)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16384,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    results = []
    for out in outputs:
        generated = out[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        text = strip_think_tags(text)
        for prefix in ("assistant\n", "user\n", "system\n"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        results.append(text)
    return results
