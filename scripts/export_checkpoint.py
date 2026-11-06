"""Export a training checkpoint as a compact, model-only file for the repository.

  python scripts/export_checkpoint.py English/checkpoints/best.pth.tar English/checkpoints/final_model.pth.tar

Weight matrices are stored in fp16 (~70 MB, below GitHub's 100 MB limit); position tables,
bins and 1-D tensors stay fp32. Loading into the fp32 model converts back automatically.
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)
state = ckpt["model"]
out = {}
for k, v in state.items():
    keep32 = v.ndim < 2 or "position_enc" in k or "bins" in k or not v.is_floating_point()
    out[k] = v if keep32 else v.half()
torch.save({"model": out, "step": ckpt.get("step")}, dst)
import os

print("wrote %s (%.1f MB), %d tensors" % (dst, os.path.getsize(dst) / 1e6, len(out)))
