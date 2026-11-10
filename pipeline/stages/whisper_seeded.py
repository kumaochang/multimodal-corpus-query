"""Run MOCO's make_modern_pipeline.py unchanged, with the random seed reset before Whisper decodes
each clip (PREREGISTRATION.md Amendment 8 item 1).

Whisper's temperature fallback samples, so two unseeded runs differ. This wrapper patches
whisper.model.Whisper.transcribe to call torch.manual_seed(SEED) (which also seeds CUDA), and to
seed Python's and NumPy's generators, before every call, then runs the script as __main__. The
script file is not modified; its arguments are passed through. Per-clip seeding makes a resumed
run identical to an uninterrupted one, whatever order glob returns the files in.

Usage (from the MOCO working folder, e.g. work/):
  HF_TOKEN="$(cat ~/.hf_token)" python ~/dissertation/pipeline/stages/whisper_seeded.py \
      ./make_modern_pipeline.py [script args]
"""
import random, runpy, sys
import numpy as np
import torch
import whisper.model

SEED = 20261012
_orig = whisper.model.Whisper.transcribe

def _seeded(self, *a, **k):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    return _orig(self, *a, **k)

whisper.model.Whisper.transcribe = _seeded

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    print(f"[whisper_seeded] seed {SEED} before each transcribe; running {script}", flush=True)
    runpy.run_path(script, run_name="__main__")
