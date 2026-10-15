from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

ASSET_MANIFEST = os.environ.get("DIA2_ASSET_MANIFEST", "dia2_assets.json")


@dataclass(frozen=True)
class AssetBundle:
    config_path: str
    weights_path: str
    tokenizer_id: Optional[str]
    mimi_id: Optional[str]
    repo_id: Optional[str]


def resolve_assets(
    *,
    repo: Optional[str],
    config_path: Optional[str | Path],
    weights_path: Optional[str | Path],
    manifest_name: Optional[str] = None,
) -> AssetBundle:
    repo_id = repo
    manifest_name = manifest_name or ASSET_MANIFEST
    if repo_id and (config_path or weights_path):
        raise ValueError("Provide either repo or config+weights, not both")
    if config_path is None or weights_path is None:
        if repo_id is None:
            raise ValueError("Must specify repo or config+weights")
        manifest = load_manifest(repo_id, manifest_name)
        config_name = manifest.get("config", "config.json")
        weights_name = manifest.get("weights", "model.safetensors")
        config_local = hf_hub_download(repo_id, config_name)
        weights_local = hf_hub_download(repo_id, weights_name)
        return AssetBundle(
            config_path=config_local,
            weights_path=weights_local,
            tokenizer_id=manifest.get("tokenizer") or repo_id,
            mimi_id=manifest.get("mimi"),
            repo_id=repo_id,
        )
    return AssetBundle(str(config_path), str(weights_path), None, None, repo_id)


def load_manifest(repo_id: str, manifest_name: str) -> dict:
    if not manifest_name:
        return {}
    try:
        path = hf_hub_download(repo_id, manifest_name)
    except Exception:
        return {}
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        return {}


__all__ = ["AssetBundle", "ASSET_MANIFEST", "resolve_assets", "load_manifest"]
