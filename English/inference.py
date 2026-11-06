"""English (Svarah): synthesise the five assignment test sentences and plot the variance adaptor.

  python inference.py                                    # uses checkpoints/final_model.pth.tar
  python inference.py --ckpt_file checkpoints/best.pth.tar
  python inference.py --restore_step 40000               # checkpoints/40000.pth.tar

Outputs (in Results/English/): test_generated_audios/*.wav,
variance_adaptor_plots/{duration,pitch,energy}/<id>.png (+ .csv with the numbers),
test_summary.json
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ["FS2_LANG"] = "en"
sys.path.insert(0, os.path.join(ROOT, "common"))

from utils.config import load_config  # noqa: E402
from inference_core import run_inference  # noqa: E402
from text.english import english_to_phone_string as frontend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--ckpt_file", default=None)
    ap.add_argument("--restore_step", type=int, default=0)
    ap.add_argument("--speaker", default=None, help="speaker name/id (multi-speaker models); default: auto")
    ap.add_argument("--test_sentences", default=os.path.join(ROOT, "test_sentences.json"))
    ap.add_argument("--results_dir", default=os.path.join(ROOT, "Results", "English"))
    ap.add_argument("--pitch_control", type=float, default=1.0)
    ap.add_argument("--energy_control", type=float, default=1.0)
    ap.add_argument("--duration_control", type=float, default=1.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not args.ckpt_file and not args.restore_step:
        default = os.path.join(cfg["train"]["path"]["ckpt_path"], "final_model.pth.tar")
        if not os.path.exists(default):
            default = os.path.join(cfg["train"]["path"]["ckpt_path"], "best.pth.tar")
        args.ckpt_file = default
    run_inference(cfg, args, "English", frontend, args.results_dir, args.test_sentences)


if __name__ == "__main__":
    main()
