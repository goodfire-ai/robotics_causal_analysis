"""Counterfactual scenes, donor-recipient pairs, and token positions (data/manifest.json)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ("molmoact2", "pi05", "molmobot")
N_FAMILIES = 13


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(data_dir: Path = DATA) -> dict:
    """Load the manifest and resolve image paths to absolute paths.

    Structure:
      families[i]    = {index, name, scenes, observations, pairs: {"A"|"B"|"C": [[donor, recipient], ...]}}
      observations[ep] = {scene, prompt, state, other, images: {ext, wrist}, image_sha256}
                         (`other` = the same scene with the instruction for the other object)
      models[name]   = {n_layers, residual_layers, observations[ep]: {groups, token_ids|lang_ids, ...}}
    """
    data_dir = Path(data_dir)
    m = json.loads((data_dir / "manifest.json").read_text())
    for o in m["observations"].values():
        o["images"] = {k: str(data_dir / p) for k, p in o["images"].items()}
    return m


def task_spec(task: int) -> tuple[str, int, int]:
    """Map a flat task index (0..311) to (model, family, seed): 3 models x 13 families x 8 seeds."""
    return MODELS[task % 3], (task // 3) % N_FAMILIES, 1000 + task // (3 * N_FAMILIES)
