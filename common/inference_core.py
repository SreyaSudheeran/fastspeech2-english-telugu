"""Synthesis of the assignment test sentences + variance-adaptor plots."""
import json
import os

import numpy as np
import torch
from scipy.io import wavfile

from text import _symbol_to_id, text_to_sequence
from utils.config import split_config
from utils.model import get_model, get_vocoder, vocoder_infer
from utils.plotting import plot_contour, plot_duration, safe_title, write_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_speaker(preprocess_config, model_config, requested):
    if not model_config.get("multi_speaker"):
        return 0, None
    pre = preprocess_config["path"]["preprocessed_path"]
    with open(os.path.join(pre, "speakers.json")) as f:
        spk_map = json.load(f)
    if requested not in (None, "auto"):
        name = requested if isinstance(requested, str) and requested in spk_map else None
        if name is None:
            return int(requested), None
        return spk_map[name], name
    # "auto": the training speaker with the most audio
    tot = {}
    with open(os.path.join(pre, "train.txt"), encoding="utf-8") as f:
        for line in f:
            b, s, *_ = line.strip().split("|")
            tot[s] = tot.get(s, 0) + 1
    best = max(tot, key=tot.get)
    return spk_map[best], best


def run_inference(cfg, args, language, frontend, results_dir, test_sentences_path):
    preprocess_config, model_config, train_config = split_config(cfg)
    inf = cfg.get("inference", {})
    pre = preprocess_config["path"]["preprocessed_path"]
    hop = preprocess_config["preprocessing"]["stft"]["hop_length"]
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    ms_per_frame = 1000.0 * hop / sr
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    with open(os.path.join(pre, "stats.json")) as f:
        stats = json.load(f)
    p_mean, p_std = stats["pitch"][2], stats["pitch"][3]
    e_mean, e_std = stats["energy"][2], stats["energy"][3]

    model = get_model(args, (preprocess_config, model_config, train_config), device, train=False).to(device)
    vocoder = get_vocoder(model_config, device)
    speaker_id, speaker_name = pick_speaker(preprocess_config, model_config, args.speaker or inf.get("speaker"))
    print("Speaker id:", speaker_id, speaker_name or "")

    audio_dir = os.path.join(results_dir, "test_generated_audios")
    plot_dir = os.path.join(results_dir, "variance_adaptor_plots")
    for sub in ("duration", "pitch", "energy"):
        os.makedirs(os.path.join(plot_dir, sub), exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    with open(test_sentences_path, encoding="utf-8") as f:
        sentences = json.load(f)[language]

    summary = []
    for item in sentences:
        sid, text = item["id"], item["text"]
        phone_str = frontend(text)
        tokens = phone_str.strip("{}").split()
        kept = [t for t in tokens if ("@" + t) in _symbol_to_id]
        seq = np.array(text_to_sequence(phone_str, cleaners))
        assert len(seq) == len(kept), "phone/symbol mismatch for " + sid
        print("[%s] %s\n     %s" % (sid, text, phone_str))

        texts = torch.from_numpy(seq).long().unsqueeze(0).to(device)
        src_lens = torch.tensor([len(seq)]).to(device)
        speakers = torch.tensor([speaker_id]).long().to(device)
        with torch.no_grad():
            out = model(
                speakers, texts, src_lens, len(seq),
                p_control=args.pitch_control, e_control=args.energy_control, d_control=args.duration_control,
            )
            mel = out[1]
            mel_len = int(out[9][0].item())
            wav = vocoder_infer(mel.transpose(1, 2), vocoder, model_config, preprocess_config,
                                lengths=out[9] * hop)[0]
        wavfile.write(os.path.join(audio_dir, sid + ".wav"), sr, wav)

        dur = out[5][0, : len(seq)].cpu().numpy()                     # rounded frames
        log_dur = out[4][0, : len(seq)].cpu().numpy()                 # raw log(d+1)
        pitch_hz = out[2][0, : len(seq)].cpu().numpy() * p_std + p_mean
        energy = out[3][0, : len(seq)].cpu().numpy() * e_std + e_mean
        title = safe_title(sid, text)

        plot_duration(os.path.join(plot_dir, "duration", sid + ".png"), kept, dur, ms_per_frame, title)
        plot_contour(os.path.join(plot_dir, "pitch", sid + ".png"), kept, dur, pitch_hz, ms_per_frame,
                     "predicted F0 (Hz)", title + "\npredicted phoneme-level pitch (F0)", "tomato", ref=p_mean)
        plot_contour(os.path.join(plot_dir, "energy", sid + ".png"), kept, dur, energy, ms_per_frame,
                     "predicted energy (STFT-norm)", title + "\npredicted phoneme-level energy", "darkviolet", ref=e_mean)
        write_csv(os.path.join(plot_dir, "duration", sid + ".csv"), ["phone", "frames", "ms", "log_dur_pred"],
                  [(p, int(d), round(float(d) * ms_per_frame, 1), round(float(l), 4)) for p, d, l in zip(kept, dur, log_dur)])
        write_csv(os.path.join(plot_dir, "pitch", sid + ".csv"), ["phone", "f0_hz"],
                  [(p, round(float(v), 2)) for p, v in zip(kept, pitch_hz)])
        write_csv(os.path.join(plot_dir, "energy", sid + ".csv"), ["phone", "energy"],
                  [(p, round(float(v), 4)) for p, v in zip(kept, energy)])

        summary.append({
            "id": sid, "text": text, "phones": phone_str, "n_phones": len(seq),
            "duration_s": round(mel_len * ms_per_frame / 1000, 3),
            "mean_f0_hz": round(float(pitch_hz.mean()), 1),
            "f0_range_hz": [round(float(pitch_hz.min()), 1), round(float(pitch_hz.max()), 1)],
        })
    with open(os.path.join(results_dir, "test_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"speaker": speaker_name, "speaker_id": speaker_id, "sentences": summary}, f,
                  ensure_ascii=False, indent=2)
    print("Done. Audio ->", audio_dir, "| plots ->", plot_dir)
