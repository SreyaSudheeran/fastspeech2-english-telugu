"""Telugu (IndicTTS Telugu): preprocessing.

Stages
  prepare   select / sub-sample / split the corpus, export wav+lab, write the MFA dictionary
  align     run the Montreal Forced Aligner (trains a Telugu model on the selected data)
  features  mel / F0 / energy / duration extraction from the TextGrids -> preprocessed_data/
  all       prepare -> align -> features

Examples
  python preprocess.py --stage prepare
  python preprocess.py --stage align          # needs the `mfa` CLI (conda), else prints the commands
  python preprocess.py --stage features
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ["FS2_LANG"] = "te"
sys.path.insert(0, os.path.join(ROOT, "common"))

from utils.config import load_config  # noqa: E402


def mfa_commands(cfg, jobs):
    pp = cfg["preprocess"]
    raw, pre, lex = pp["path"]["raw_path"], pp["path"]["preprocessed_path"], pp["path"]["lexicon_path"]
    tg = os.path.join(pre, "TextGrid")
    # no pretrained Telugu aligner is assumed: train a small one on the selected data
    cmds = [
        ["mfa", "train", raw, lex, tg, "--clean", "--single_speaker",
         "--output_model_path", os.path.join(pre, "telugu_mfa_acoustic.zip"), "-j", str(jobs)],
    ]
    return cmds


def run_align(cfg, jobs):
    cmds = mfa_commands(cfg, jobs)
    if shutil.which("mfa") is None:
        print("The `mfa` command was not found. Install it (conda install -c conda-forge montreal-forced-aligner)")
        print("and run these commands, then continue with --stage features:\n")
        for c in cmds:
            print("  " + " ".join(c))
        sys.exit(2)
    for c in cmds:
        print("$", " ".join(c))
        subprocess.run(c, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--stage", choices=["prepare", "align", "features", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=2, help="MFA worker processes")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.stage in ("prepare", "all"):
        from preprocessor.data_prep import prepare

        prepare(cfg, "te")
    if args.stage in ("align", "all"):
        run_align(cfg, args.jobs)
    if args.stage in ("features", "all"):
        from preprocessor.preprocessor import Preprocessor

        sel = cfg["preprocess"]["selection"]
        Preprocessor(
            cfg["preprocess"],
            max_seq_len=cfg["model"]["max_seq_len"],
            train_hours=sel["train_hours"],
            val_hours=sel["val_hours"],
        ).build_from_path()


if __name__ == "__main__":
    main()
