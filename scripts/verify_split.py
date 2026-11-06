"""Verify that train and validation are disjoint and report the final data amounts.

  python scripts/verify_split.py English     (or Telugu)

Checks: no shared utterance id, no shared transcript, no byte-identical audio.
Writes Results/<Language>/data_verification.json.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "common"))
from utils.config import load_config  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("language", choices=["English", "Telugu"])
ap.add_argument("--config", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
lang = a.language
cfg = load_config(a.config or os.path.join(ROOT, lang, "config.yaml"))
pre = cfg["preprocess"]["path"]["preprocessed_path"]
raw = cfg["preprocess"]["path"]["raw_path"]


def read(name):
    rows = []
    with open(os.path.join(pre, name + ".txt"), encoding="utf-8") as f:
        for line in f:
            b, s, ph, txt = line.rstrip("\n").split("|")
            rows.append((b, s, txt))
    return rows


res = {}
data = {n: read(n) for n in ("train", "val")}
hashes = {}
for n, rows in data.items():
    hashes[n] = {
        hashlib.sha1(open(os.path.join(raw, s, b + ".wav"), "rb").read()[44:]).hexdigest() for b, s, _ in rows
    }
res["shared_ids"] = len({r[0] for r in data["train"]} & {r[0] for r in data["val"]})
res["shared_transcripts"] = len({r[2] for r in data["train"]} & {r[2] for r in data["val"]})
res["shared_audio_files"] = len(hashes["train"] & hashes["val"])
res["val_speakers_not_in_train"] = len({r[1] for r in data["val"]} - {r[1] for r in data["train"]})
with open(os.path.join(pre, "data_summary.json")) as f:
    res["data_summary"] = json.load(f)
sel = os.path.join(pre, "selection_report.json")
if os.path.exists(sel):
    with open(sel) as f:
        res["selection_report"] = json.load(f)
ok = res["shared_ids"] == res["shared_transcripts"] == res["shared_audio_files"] == 0
res["no_overlap"] = ok
out = a.out or os.path.join(ROOT, "Results", lang, "data_verification.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k not in ("selection_report",)}, indent=2))
print("NO OVERLAP" if ok else "OVERLAP FOUND")
sys.exit(0 if ok else 1)
