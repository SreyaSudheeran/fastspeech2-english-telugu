"""Training loop (adapted from ming024/FastSpeech2 train.py).

Changes: single combined config; CSV loss logs (interval-averaged training loss, every
validation loss) used to draw training_loss.png / validation_loss.png; best-validation
checkpoint; --resume; DataLoader workers; loss plots written at the end of training.
"""
import csv
import glob
import os
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as _e:  # e.g. TensorBoard pulling in a TensorFlow/JAX that clashes with numpy 1.26
    print("TensorBoard unavailable (%s) - continuing with CSV logs only" % type(_e).__name__)

    class SummaryWriter:  # no-op stand-in; loss plots come from the CSV logs
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: None
from tqdm import tqdm

from dataset import Dataset
from evaluate import evaluate
from model import FastSpeech2Loss
from utils.model import get_model, get_param_num, get_vocoder
from utils.plotting import LOSS_COLS, plot_loss_curves, plot_train_vs_val
from utils.tools import log, synth_one_sample, to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def latest_step(ckpt_dir):
    steps = [
        int(m.group(1))
        for f in glob.glob(os.path.join(ckpt_dir, "*.pth.tar"))
        if (m := re.match(r"(\d+)\.pth\.tar$", os.path.basename(f)))
    ]
    return max(steps) if steps else 0


def _append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step"] + LOSS_COLS)
        w.writerow(row)


def run_training(args, cfg, results_dir):
    preprocess_config, model_config, train_config = (
        cfg["preprocess"], cfg["model"], cfg["train"],
    )
    configs = (preprocess_config, model_config, train_config)
    if args.resume:
        args.restore_step = latest_step(train_config["path"]["ckpt_path"])
        print("Resuming from step", args.restore_step)

    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    log_dir = train_config["path"]["log_path"]
    train_csv, val_csv = os.path.join(log_dir, "train_losses.csv"), os.path.join(log_dir, "val_losses.csv")
    if not args.restore_step:  # fresh run -> fresh logs
        for f in (train_csv, val_csv):
            if os.path.exists(f):
                os.remove(f)

    total_step = train_config["step"]["total_step"]
    if args.restore_step >= total_step:
        print("Already trained for %d >= total_step %d steps; redrawing plots only." % (args.restore_step, total_step))
        finish(train_csv, val_csv, results_dir, train_config)
        return

    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    assert batch_size * group_size < len(dataset), "dataset too small for batch size"
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=train_config.get("num_workers", 2),
    )

    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    print("Number of FastSpeech2 parameters:", get_param_num(model))
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    vocoder = get_vocoder(model_config, device)

    train_logger = SummaryWriter(os.path.join(log_dir, "train"))
    val_logger = SummaryWriter(os.path.join(log_dir, "val"))

    step = args.restore_step + 1
    epoch = 1
    o, s = train_config["optimizer"], train_config["step"]
    grad_acc_step, clip = o["grad_acc_step"], o["grad_clip_thresh"]
    total_step, log_step, save_step = s["total_step"], s["log_step"], s["save_step"]
    synth_step, val_step = s["synth_step"], s["val_step"]
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    best_val = float("inf")
    if args.restore_step and os.path.exists(val_csv):  # keep best.pth.tar meaningful after a resume
        with open(val_csv) as f:
            vals = [float(r["total"]) for r in csv.DictReader(f)]
        best_val = min(vals) if vals else best_val
    interval = [0.0] * 6
    n_interval = 0

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    def save(name, with_opt=True):
        ckpt = {"model": model.module.state_dict(), "step": step_now}
        if with_opt:
            ckpt["optimizer"] = optimizer._optimizer.state_dict()
        torch.save(ckpt, os.path.join(train_config["path"]["ckpt_path"], name))

    step_now = step
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                step_now = step
                batch = to_device(batch, device)
                output = model(*(batch[2:]))
                losses = Loss(batch, output)
                (losses[0] / grad_acc_step).backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                vals = [l.item() for l in losses]
                interval = [a + b for a, b in zip(interval, vals)]
                n_interval += 1

                if step % log_step == 0:
                    avg = [a / n_interval for a in interval]
                    interval, n_interval = [0.0] * 6, 0
                    msg = "Step {}/{}, Total: {:.4f}, Mel: {:.4f}, PostNet: {:.4f}, Pitch: {:.4f}, Energy: {:.4f}, Duration: {:.4f}".format(step, total_step, *avg)
                    with open(os.path.join(log_dir, "train_log.txt"), "a") as f:
                        f.write(msg + "\n")
                    outer_bar.write(msg)
                    _append_csv(train_csv, [step] + avg)
                    log(train_logger, step, losses=avg)

                if step % synth_step == 0:
                    fig, wav_rec, wav_pred, tag = synth_one_sample(batch, output, vocoder, model_config, preprocess_config)
                    log(train_logger, fig=fig, tag="Training/step_{}_{}".format(step, tag))
                    log(train_logger, audio=wav_rec, sampling_rate=sr, tag="Training/step_{}_{}_reconstructed".format(step, tag))
                    log(train_logger, audio=wav_pred, sampling_rate=sr, tag="Training/step_{}_{}_synthesized".format(step, tag))

                if step % val_step == 0:
                    model.eval()
                    msg, means = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(log_dir, "val_log.txt"), "a") as f:
                        f.write(msg + "\n")
                    outer_bar.write(msg)
                    _append_csv(val_csv, [step] + list(means))
                    if means[0] < best_val:
                        best_val = means[0]
                        save("best.pth.tar", with_opt=False)
                    model.train()

                if step % save_step == 0:
                    save("{}.pth.tar".format(step))

                if step == total_step:
                    save("final.pth.tar")
                    finish(train_csv, val_csv, results_dir, train_config)
                    return
                step += 1
                outer_bar.update(1)
            inner_bar.update(1)
        epoch += 1


def finish(train_csv, val_csv, results_dir, train_config):
    os.makedirs(results_dir, exist_ok=True)
    try:
        plot_loss_curves(train_csv, os.path.join(results_dir, "training_loss.png"), "Training", smooth=5)
        plot_loss_curves(val_csv, os.path.join(results_dir, "validation_loss.png"), "Validation")
        plot_train_vs_val(train_csv, val_csv, os.path.join(results_dir, "train_vs_val_total_loss.png"),
                          "Train vs validation total loss")
        print("Loss plots written to", results_dir)
    except Exception as e:  # never lose a finished training run over a plot
        print("Could not draw loss plots:", e)
