import os
import json
import zipfile

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim

_HIFIGAN_DIR = os.path.dirname(os.path.abspath(hifigan.__file__))


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt = None
    ckpt_file = getattr(args, "ckpt_file", None)
    if ckpt_file:
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    elif args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step and ckpt is not None and "optimizer" in ckpt:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def _hifigan_ckpt(name):
    """Path to a HiFi-GAN generator checkpoint; unzips the bundled .zip on first use."""
    path = os.path.join(_HIFIGAN_DIR, "generator_{}.pth.tar".format(name))
    if not os.path.exists(path):
        zpath = path + ".zip"
        if not os.path.exists(zpath):
            raise FileNotFoundError(
                "Missing HiFi-GAN checkpoint {} (and {})".format(path, zpath)
            )
        with zipfile.ZipFile(zpath) as z:
            z.extractall(_HIFIGAN_DIR)
    return path


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open(os.path.join(_HIFIGAN_DIR, "config.json"), "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        ckpt = torch.load(
            _hifigan_ckpt(speaker), map_location=device, weights_only=False
        )
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
