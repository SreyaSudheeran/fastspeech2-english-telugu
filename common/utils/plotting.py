"""Plots: loss curves and variance-adaptor (duration / pitch / energy) predictions."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOSS_COLS = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def _setup_fonts():
    """Use a Telugu-capable font for titles when one is installed; otherwise titles
    fall back to ASCII (phone labels on the axes are always ASCII)."""
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Noto Sans Telugu", "Noto Serif Telugu", "Lohit Telugu", "Gautami", "Nirmala UI", "Pothana2000", "Vemana2000"]:
        if cand in names:
            matplotlib.rcParams["font.sans-serif"].insert(0, cand)
            return True
    return False


HAS_TELUGU_FONT = _setup_fonts()


def safe_title(sid, text, n_chars=70):
    """Sentence id + text; text dropped when it has Telugu script and no font for it."""
    if any("\u0C00" <= c <= "\u0C7F" for c in text) and not HAS_TELUGU_FONT:
        return sid
    return "%s  |  %s" % (sid, text if len(text) <= n_chars else text[: n_chars - 3] + "...")


def read_loss_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    steps = np.array([int(r["step"]) for r in rows])
    return steps, {c: np.array([float(r[c]) for r in rows]) for c in LOSS_COLS}


def _smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    ker = np.ones(k) / k
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, ker, mode="valid")


def plot_loss_curves(csv_path, out_png, title, smooth=1):
    steps, d = read_loss_csv(csv_path)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(steps, _smooth(d["total"], smooth), color="C0")
    ax[0].set_title(title + " - total loss")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("loss")
    ax[0].grid(alpha=0.3)
    for c in LOSS_COLS[1:]:
        ax[1].plot(steps, _smooth(d[c], smooth), label=c)
    ax[1].set_yscale("log")
    ax[1].set_title(title + " - components (log scale)")
    ax[1].set_xlabel("step")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_train_vs_val(train_csv, val_csv, out_png, title):
    s1, d1 = read_loss_csv(train_csv)
    s2, d2 = read_loss_csv(val_csv)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s1, _smooth(d1["total"], 10), label="train (smoothed)")
    ax.plot(s2, d2["total"], label="validation", marker="o", ms=3)
    ax.set_xlabel("step")
    ax.set_ylabel("total loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _boundaries(durations):
    edges = np.concatenate([[0], np.cumsum(durations)])
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers


def _phone_ticks(ax, centers, labels):
    ax.set_xticks(centers)
    ax.set_xticklabels([("·" if l in ("sp", "spn", "sil") else l) for l in labels],
                       rotation=90, fontsize=7)


def plot_duration(path, phones, dur_frames, ms_per_frame, title):
    ms = np.asarray(dur_frames) * ms_per_frame
    fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(phones)), 4))
    colors = ["#bbbbbb" if p in ("sp", "spn", "sil") else "C0" for p in phones]
    ax.bar(np.arange(len(phones)), ms, color=colors)
    ax.set_xticks(np.arange(len(phones)))
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_ylabel("predicted duration (ms)")
    ax.set_title("%s\ntotal %.2f s, %d phones (grey = pause)" % (title, ms.sum() / 1000, len(phones)),
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_contour(path, phones, dur_frames, phone_values, ms_per_frame, ylabel, title, color, ref=None):
    """Phoneme-level values expanded to frames (as the decoder sees them)."""
    d = np.maximum(np.asarray(dur_frames).astype(int), 0)
    frame_vals = np.repeat(phone_values, d)
    edges, centers = _boundaries(d)
    fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(phones)), 4))
    ax.step(np.arange(len(frame_vals)), frame_vals, where="post", color=color, lw=1.5)
    ax.plot(centers, phone_values, "o", color=color, ms=3)
    for e in edges:
        ax.axvline(e, color="k", alpha=0.08, lw=0.6)
    if ref is not None:
        ax.axhline(ref, color="grey", ls="--", lw=1, label="training mean")
        ax.legend(fontsize=8)
    _phone_ticks(ax, centers, phones)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("frame (%.1f ms) / phone" % ms_per_frame)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
