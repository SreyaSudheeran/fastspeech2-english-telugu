"""Dataset selection, sub-sampling, train/val split, raw-corpus export and MFA dictionary.

Pipeline (see English/preprocess.py, Telugu/preprocess.py):
  1. read metadata (HuggingFace dataset or a local manifest)
  2. clean transcripts per language, drop unusable rows, de-duplicate transcripts
  3. filter by duration, pick the speaker set
  4. round-robin sub-sampling across speakers up to (train+val) hours
  5. utterance-level stratified split into train / val
  6. decode + resample + write wav/lab only for selected utterances
  7. write the pronunciation dictionary for the Montreal Forced Aligner
"""
import hashlib
import io
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text.cleaners import english_cleaners
from text.telugu import normalize_telugu, telugu_words, word_to_phones

_TELUGU_CHARS = re.compile(r"^[ఀ-౿\s,.?!\-:]+$")
_ENGLISH_CHARS = re.compile(r"^[a-z' ,.?!]+$")


# ----------------------------------------------------------------------------
# Transcript cleaning
# ----------------------------------------------------------------------------
def clean_transcript(language, text):
    """-> list of words, or None when the transcript is unusable for TTS."""
    if not isinstance(text, str) or not text.strip():
        return None
    if language == "en":
        if re.search(r"[\[\]<>()*#@_/\\|{}=+~^%]", text):  # tags / annotations
            return None
        t = english_cleaners(text)
        if not _ENGLISH_CHARS.match(t):
            return None
        words = re.sub(r"[^a-z' ]", " ", t).split()
        words = [re.sub(r"^'+|'+$", "", w) for w in words]
        words = [w for w in words if re.search("[a-z]", w)]
    else:
        t = normalize_telugu(text)
        if not _TELUGU_CHARS.match(t):
            return None
        words = telugu_words(text)
    return words if len(words) >= 2 else None


# ----------------------------------------------------------------------------
# Metadata readers
# ----------------------------------------------------------------------------
class Source:
    """Uniform access to rows (metadata) and audio of a corpus."""

    def __init__(self, spec, language):
        self.spec = spec
        self.language = language
        self.kind = spec["source"]  # "hf" | "manifest"
        if self.kind == "hf":
            from datasets import Audio, load_dataset

            ds = load_dataset(spec["name"], split=spec.get("split", "train"))
            self.audio_col = spec.get("audio_column") or next(
                k for k, v in ds.features.items() if isinstance(v, Audio)
            )
            self.text_col = spec.get("text_column") or next(
                k
                for k in ["text", "transcript", "sentence", "transcription", "normalized_text"]
                if k in ds.features
            )
            self.meta = ds.remove_columns([self.audio_col])
            self.audio = ds.cast_column(self.audio_col, Audio(decode=False))
        else:
            self.rows = []
            path = spec["manifest"]
            base = os.path.dirname(os.path.abspath(path))
            with open(path, encoding="utf-8") as f:
                if path.endswith(".jsonl"):
                    raw = [json.loads(l) for l in f if l.strip()]
                else:
                    import csv

                    raw = list(csv.DictReader(f))
            for r in raw:
                ap = r["audio_filepath"]
                r["audio_filepath"] = ap if os.path.isabs(ap) else os.path.join(base, ap)
                self.rows.append(r)
            self.text_col = "text"

    def rows_meta(self):
        it = self.meta if self.kind == "hf" else self.rows
        for i, row in enumerate(it):
            yield i, row

    def read_audio(self, i):
        """-> (float32 mono array, sampling_rate)"""
        import soundfile as sf

        if self.kind == "hf":
            rec = self.audio[i][self.audio_col]
            src = io.BytesIO(rec["bytes"]) if rec.get("bytes") else rec["path"]
        else:
            src = self.rows[i]["audio_filepath"]
        y, sr = sf.read(src, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y, sr

    def duration(self, i, row):
        d = row.get("duration")
        if d not in (None, ""):
            return float(d)
        import soundfile as sf

        if self.kind == "hf":
            rec = self.audio[i][self.audio_col]
            src = io.BytesIO(rec["bytes"]) if rec.get("bytes") else rec["path"]
        else:
            src = row["audio_filepath"]
        info = sf.info(src)
        return info.frames / info.samplerate


def speaker_of(row, spec):
    """Speaker label. Svarah has no speaker id -> hash of the speaker-level metadata."""
    keys = spec.get("speaker_keys")
    if not keys:
        return spec.get("single_speaker_name", "spk0")
    vals = [str(row.get(k, "")) for k in keys]
    if keys == ["speaker_id"] or len(keys) == 1:
        return re.sub(r"[^A-Za-z0-9_\-]", "_", vals[0]) or "spk0"
    return "spk_" + hashlib.md5("|".join(vals).encode()).hexdigest()[:8]


# ----------------------------------------------------------------------------
# Selection / split
# ----------------------------------------------------------------------------
def select_and_split(items, train_hours, val_hours, seed, oversample=1.0):
    """Round-robin over speakers up to (train+val)*oversample hours, then split.

    Split is by utterance within each speaker so every validation speaker is also
    seen in training (needed by the speaker-embedding model); ids and transcripts are
    disjoint by construction (transcripts are de-duplicated before this call).
    """
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for it in items:
        by_spk[it["speaker"]].append(it)
    for v in by_spk.values():
        rng.shuffle(v)

    target = (train_hours + val_hours) * 3600.0 * oversample
    selected, total = [], 0.0
    ptr = {s: 0 for s in by_spk}
    active = sorted(by_spk)
    while total < target and active:
        rng.shuffle(active)
        nxt = []
        for s in active:
            if total >= target:
                break
            it = by_spk[s][ptr[s]]
            ptr[s] += 1
            selected.append(it)
            total += it["duration"]
            if ptr[s] < len(by_spk[s]):
                nxt.append(s)
        active = nxt

    frac = val_hours / (train_hours + val_hours)
    sel_by_spk = defaultdict(list)
    for it in selected:
        sel_by_spk[it["speaker"]].append(it)
    train, val = [], []
    for s, lst in sel_by_spk.items():
        rng.shuffle(lst)
        quota = frac * sum(x["duration"] for x in lst)
        acc = 0.0
        for k, it in enumerate(lst):
            if len(lst) >= 4 and acc < quota and k < len(lst) - 1:
                val.append(it)
                acc += it["duration"]
            else:
                train.append(it)
    return train, val


def hours(items):
    return sum(i["duration"] for i in items) / 3600.0


# ----------------------------------------------------------------------------
# Main entry: prepare()
# ----------------------------------------------------------------------------
def prepare(cfg, language):
    pp = cfg["preprocess"]
    spec = pp["source"]
    sel = pp["selection"]
    sr = pp["preprocessing"]["audio"]["sampling_rate"]
    raw_path = pp["path"]["raw_path"]
    out_dir = pp["path"]["preprocessed_path"]
    seed = sel.get("seed", 1234)
    os.makedirs(out_dir, exist_ok=True)

    src = Source(spec, language)
    report = {"seed": seed, "dropped": defaultdict(int)}

    # 1-3: metadata pass (no audio decoding when a duration column exists)
    rows = []
    for i, row in tqdm(src.rows_meta(), desc="Reading metadata"):
        words = clean_transcript(language, row.get(src.text_col))
        if words is None:
            report["dropped"]["bad_transcript"] += 1
            continue
        dur = src.duration(i, row)
        if not (sel["min_duration"] <= dur <= sel["max_duration"]):
            report["dropped"]["duration_out_of_range"] += 1
            continue
        rows.append(
            {
                "index": i,
                "speaker": speaker_of(row, spec),
                "duration": dur,
                "words": words,
                "text": " ".join(words),
            }
        )
    report["after_filtering"] = {"utterances": len(rows), "hours": hours(rows)}

    # speaker policy
    if sel.get("speaker_mode") == "single_best":
        tot = defaultdict(float)
        for r in rows:
            tot[r["speaker"]] += r["duration"]
        best = max(tot, key=tot.get)
        report["chosen_speaker"] = best
        report["speaker_hours_available"] = {k: v / 3600 for k, v in tot.items()}
        rows = [r for r in rows if r["speaker"] == best]

    # de-duplicate transcripts (keeps one random utterance per sentence) so that
    # train and validation share no sentence
    if sel.get("dedupe_transcripts", True):
        rng = random.Random(seed)
        rng.shuffle(rows)
        seen, uniq = set(), []
        for r in rows:
            if r["text"] in seen:
                report["dropped"]["duplicate_transcript"] += 1
                continue
            seen.add(r["text"])
            uniq.append(r)
        rows = uniq
    report["after_dedupe"] = {"utterances": len(rows), "hours": hours(rows)}

    train, val = select_and_split(
        rows,
        sel["train_hours"],
        sel["val_hours"],
        seed,
        sel.get("oversample", 1.0),
    )
    assert not ({r["index"] for r in train} & {r["index"] for r in val}), "split overlap"
    assert not ({r["text"] for r in train} & {r["text"] for r in val}), "text overlap"

    # 6: export wav + lab for the selected utterances
    hashes = {"train": set(), "val": set()}
    vocab = set()
    split_lines = {"train": [], "val": []}
    for name, items in (("train", train), ("val", val)):
        for r in tqdm(items, desc="Exporting " + name):
            y, sr0 = src.read_audio(r["index"])
            if sr0 != sr:
                import librosa

                y = librosa.resample(y, orig_sr=sr0, target_sr=sr)
            y = y / max(1e-8, float(np.max(np.abs(y)))) * 0.95
            pcm = (y * 32767).astype(np.int16)
            base = "%s_%06d" % (language, r["index"])
            d = os.path.join(raw_path, r["speaker"])
            os.makedirs(d, exist_ok=True)
            wavfile.write(os.path.join(d, base + ".wav"), sr, pcm)
            with open(os.path.join(d, base + ".lab"), "w", encoding="utf-8") as f:
                f.write(r["text"])
            hashes[name].add(hashlib.sha1(pcm.tobytes()).hexdigest())
            vocab.update(r["words"])
            split_lines[name].append("%s|%s|%.3f" % (base, r["speaker"], r["duration"]))
    assert not (hashes["train"] & hashes["val"]), "identical audio in train and val"

    os.makedirs(os.path.join(out_dir, "splits"), exist_ok=True)
    for name in ("train", "val"):
        with open(os.path.join(out_dir, "splits", name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(split_lines[name]) + "\n")

    # 7: pronunciation dictionary for MFA
    lex_path = pp["path"]["lexicon_path"]
    os.makedirs(os.path.dirname(lex_path), exist_ok=True)
    with open(lex_path, "w", encoding="utf-8") as f:
        for w in tqdm(sorted(vocab), desc="Building dictionary"):
            if language == "en":
                from text.english import word_pronunciations

                for pron in word_pronunciations(w):
                    f.write("%s\t%s\n" % (w, " ".join(pron)))
            else:
                f.write("%s\t%s\n" % (w, " ".join(word_to_phones(w))))

    report.update(
        {
            "selected_train": {"utterances": len(train), "hours": hours(train)},
            "selected_val": {"utterances": len(val), "hours": hours(val)},
            "n_speakers_train": len({r["speaker"] for r in train}),
            "n_speakers_val": len({r["speaker"] for r in val}),
            "dictionary_words": len(vocab),
            "dropped": dict(report["dropped"]),
        }
    )
    with open(os.path.join(out_dir, "selection_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report
