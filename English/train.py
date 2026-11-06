"""English (Svarah): training (FastSpeech 2).

  python train.py                 # fresh run with config.yaml
  python train.py --resume        # continue from the newest checkpoint in checkpoints/
  python train.py --restore_step 20000
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ["FS2_LANG"] = "en"
sys.path.insert(0, os.path.join(ROOT, "common"))

from utils.config import load_config  # noqa: E402
from trainer import run_training  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--restore_step", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_training(args, cfg, results_dir=os.path.join(ROOT, "Results", "English"))


if __name__ == "__main__":
    main()
