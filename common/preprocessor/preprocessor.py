"""Feature extraction (mel, F0, energy, durations) from MFA TextGrids.

Adapted from ming024/FastSpeech2 preprocessor/preprocessor.py. Changes (marked MOD):
  * train/val membership comes from splits/{train,val}.txt written by data_prep.py
    instead of a random shuffle, and pitch/energy statistics use TRAIN utterances only;
  * MFA >= 2 writes silences as empty intervals ("") instead of sil/sp/spn -> mapped to "sp";
  * wav is loaded at the configured sampling rate; utterances longer than max_seq_len
    frames are skipped; train / val are trimmed to the requested hours;
  * a bug in the upstream loop that reused stale pitch/energy of a previous
    utterance when a TextGrid was missing is fixed.
"""
import json
import os
import random
import shutil

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config, max_seq_len, train_hours=None, val_hours=None):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.max_seq_len = max_seq_len
        self.train_hours = train_hours
        self.val_hours = val_hours
        self.seed = 1234

        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )
        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    # ------------------------------------------------------------------
    def _read_split(self, name):
        path = os.path.join(self.out_dir, "splits", name + ".txt")
        with open(path, encoding="utf-8") as f:
            return [l.strip().split("|")[:2] for l in f if l.strip()]  # (basename, speaker)

    def _textgrid_path(self, speaker, basename):
        p = os.path.join(self.out_dir, "TextGrid", speaker, basename + ".TextGrid")
        if os.path.exists(p):
            return p
        p2 = os.path.join(self.out_dir, "TextGrid", basename + ".TextGrid")
        return p2 if os.path.exists(p2) else None

    def build_from_path(self):
        for sub in ["mel", "pitch", "energy", "duration"]:
            # MOD: start clean so a re-run cannot normalise stale files twice
            shutil.rmtree(os.path.join(self.out_dir, sub), ignore_errors=True)
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)

        splits = {"train": self._read_split("train"), "val": self._read_split("val")}
        pitch_scaler, energy_scaler = StandardScaler(), StandardScaler()
        out = {"train": [], "val": []}
        frames = {"train": {}, "val": {}}
        n_missing = n_failed = n_long = 0
        speakers = {}

        for name in ("train", "val"):
            for basename, speaker in tqdm(splits[name], desc="Features/" + name):
                speakers.setdefault(speaker, len(speakers))
                tg_path = self._textgrid_path(speaker, basename)
                if tg_path is None:
                    n_missing += 1  # MFA could not align this utterance
                    continue
                ret = self.process_utterance(speaker, basename, tg_path)
                if ret == "long":
                    n_long += 1
                    continue
                if ret is None:
                    n_failed += 1
                    continue
                info, pitch, energy, n = ret
                out[name].append((info, n))
                frames[name][info.split("|")[0]] = n
                if name == "train":  # MOD: statistics from training data only
                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

        # MOD: trim to requested hours (alignment failures are over-sampled for)
        rng = random.Random(self.seed)
        for name, target_h in (("train", self.train_hours), ("val", self.val_hours)):
            if target_h:
                rng.shuffle(out[name])
                kept, tot = [], 0.0
                for info, n in out[name]:
                    if tot >= target_h * 3600:
                        break
                    kept.append((info, n))
                    tot += n * self.hop_length / self.sampling_rate
                out[name] = kept

        if self.pitch_normalization:
            pitch_mean, pitch_std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
        else:
            pitch_mean, pitch_std = 0, 1
        if self.energy_normalization:
            energy_mean, energy_std = energy_scaler.mean_[0], energy_scaler.scale_[0]
        else:
            energy_mean, energy_std = 0, 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump(speakers, f)
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            json.dump(
                {
                    "pitch": [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)],
                    "energy": [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)],
                },
                f,
            )

        summary = {}
        for name in ("train", "val"):
            with open(os.path.join(self.out_dir, name + ".txt"), "w", encoding="utf-8") as f:
                for info, _ in out[name]:
                    f.write(info + "\n")
            secs = sum(n for _, n in out[name]) * self.hop_length / self.sampling_rate
            summary[name] = {
                "utterances": len(out[name]),
                "hours": round(secs / 3600, 3),
                "speakers": len({i.split("|")[1] for i, _ in out[name]}),
            }
        summary.update(
            {
                "skipped_no_textgrid": n_missing,
                "skipped_alignment_failed": n_failed,
                "skipped_too_long": n_long,
                "pitch_mean_std_hz": [float(pitch_mean), float(pitch_std)],
            }
        )
        with open(os.path.join(self.out_dir, "data_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return summary

    # ------------------------------------------------------------------
    def process_utterance(self, speaker, basename, tg_path):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))

        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)  # MOD: explicit sr
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.readline().strip("\n")

        if sum(duration) > self.max_seq_len:  # MOD
            return "long"

        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))
            pos = 0
            for i, d in enumerate(duration):
                pitch[i] = np.mean(pitch[pos : pos + d]) if d > 0 else 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            pos = 0
            for i, d in enumerate(duration):
                energy[i] = np.mean(energy[pos : pos + d]) if d > 0 else 0
                pos += d
            energy = energy[: len(duration)]

        np.save(os.path.join(self.out_dir, "duration", "{}-duration-{}.npy".format(speaker, basename)), duration)
        np.save(os.path.join(self.out_dir, "pitch", "{}-pitch-{}.npy".format(speaker, basename)), pitch)
        np.save(os.path.join(self.out_dir, "energy", "{}-energy-{}.npy".format(speaker, basename)), energy)
        np.save(os.path.join(self.out_dir, "mel", "{}-mel-{}.npy".format(speaker, basename)), mel_spectrogram.T)

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones, durations = [], []
        start_time = end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            if p == "":  # MOD: MFA >= 2 marks silence with an empty label
                p = "sp"

            if phones == []:  # trim leading silences
                if p in sil_phones:
                    continue
                start_time = s

            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        phones = phones[:end_idx]  # trim trailing silences
        durations = durations[:end_idx]
        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25, p75 = np.percentile(values, 25), np.percentile(values, 75)
        lower, upper = p25 - 1.5 * (p75 - p25), p75 + 1.5 * (p75 - p25)
        return values[np.logical_and(values > lower, values < upper)]

    def normalize(self, in_dir, mean, std):
        """Normalise every stored file (train AND val) with the train statistics."""
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)
            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))
        return min_value, max_value
