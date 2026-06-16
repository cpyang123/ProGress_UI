"""
regenerate_processed.py – Trigger SchenkerDiff dataset preprocessing.

Just constructs SchenkerGraphDataModule, which invokes
SchenkerDiffHeteroGraphData.__init__ → PyG Dataset._process()
→ self.process(), writing all <idx>_processed.pt files to disk.

No training, no GPU, no checkpoint writes.
"""

import os
import sys
import time
from pathlib import Path

SCHENKER_ROOT = Path("/mnt/c/Users/Peter/OneDrive/Desktop/Duke/Music ML Research/Diffusion/SchenkerDiff")
DIFFUSION_ROOT = SCHENKER_ROOT.parent

# The dataset's process() method uses glob paths like
#   ../../../SchenkerDiff/{directory}/**/*
# That requires cwd to be three levels deep under Diffusion/ — which is what
# Hydra's default output dir layout produces.  Re-create that shape.
WORK_DIR = DIFFUSION_ROOT / "outputs" / "_regen" / "work"
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
sys.path.insert(0, str(SCHENKER_ROOT))

# Inference / training code transitively imports spectre_utils → graph_tool,
# which can fail with the libgomp mismatch.  Stub them; we only need the dataset.
import types
gt = types.ModuleType("graph_tool"); gt.all = types.ModuleType("graph_tool.all")
sys.modules.setdefault("graph_tool", gt)
sys.modules.setdefault("graph_tool.all", gt.all)

from hydra import compose, initialize_config_dir
from src.datasets.schenker_dataset import SchenkerGraphDataModule


def main() -> None:
    print(f"[regen] starting at {time.strftime('%H:%M:%S')}")
    cfg_dir = str(SCHENKER_ROOT / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(config_name="config")

    processed_dir = SCHENKER_ROOT / cfg.dataset.datadir / "processed"
    n_before = len(list(processed_dir.glob("*_processed.pt"))) if processed_dir.exists() else 0
    print(f"[regen] processed_dir={processed_dir}")
    print(f"[regen] {n_before} processed files present before regeneration")

    t0 = time.time()
    SchenkerGraphDataModule(cfg)
    elapsed = time.time() - t0

    n_after = len(list(processed_dir.glob("*_processed.pt")))
    print(f"[regen] DONE in {elapsed:.1f}s → {n_after} processed files now present")


if __name__ == "__main__":
    main()
