"""Combined-config loader.

Each language has ONE config.yaml with the sections
  preprocess / model / train / inference
(the upstream repo uses three separate yaml files). Relative paths are resolved
against the directory that holds config.yaml, so scripts work from any cwd.
"""
import os

import yaml

_PATH_SECTIONS = {
    "preprocess": ["path"],
    "train": ["path"],
    "inference": ["path"],
}


def _resolve(base, value):
    if not isinstance(value, str) or os.path.isabs(value) or ":" in value.split("/")[0]:
        return value  # absolute path or URI-like ("hf:...")
    return os.path.normpath(os.path.join(base, value))


def load_config(path):
    path = os.path.abspath(path)
    base = os.path.dirname(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    for section, keys in _PATH_SECTIONS.items():
        for key in keys:
            for k, v in cfg.get(section, {}).get(key, {}).items():
                cfg[section][key][k] = _resolve(base, v)
    src = cfg.get("preprocess", {}).get("source", {})
    if "manifest" in src:
        src["manifest"] = _resolve(base, src["manifest"])
    cfg["_config_dir"] = base
    return cfg


def split_config(cfg):
    """-> (preprocess_config, model_config, train_config) as expected by upstream code."""
    return cfg["preprocess"], cfg["model"], cfg["train"]
