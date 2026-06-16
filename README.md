---
title: progress-music-generation
emoji: 🎼
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.15.2
app_file: app.py
python_version: "3.10"
pinned: false
---

# ProGress UI

Interactive demo for the **ProGress** music generation system, combining:

- **SchenkerDiff** – discrete graph-diffusion model for Schenkerian voice-leading
- **ProGress Supplement** – rejection-sampling + phrase stitching into full compositions

This folder is **self-contained**: everything the app needs at runtime — the
`phrase_stitching` code, a trimmed copy of the SchenkerDiff model code, the
phrase library, the conditioning tensors, and the `last-v1.ckpt` checkpoint — is
vendored under [`vendor/`](vendor/). No sibling research directories are
required, so it can be deployed as a single package.

## Quickstart

```bash
cd ProGress_UI
pip install -r requirements.txt          # torch 2.8 + PyG + spaces, see notes
python app.py                            # serves http://localhost:7860
```

Python 3.10 is the validated runtime. `music21` is used only for MIDI export
(MuseScore is **not** required). The MIDI player loads from a CDN, so audio
playback needs internet access.

> Running from inside the original research tree still works: if `vendor/` is
> absent, `backend.py` falls back to the sibling `../ProGress_Supplement` and
> `../SchenkerDiff` folders. The `PROGRESS_SUPPLEMENT_DIR` /
> `PROGRESS_SCHENKER_DIR` env vars override either root.

## Deploy (Hugging Face ZeroGPU)

The Space runs on **ZeroGPU**: a GPU is attached on demand for each generation
via the `@spaces.GPU` decorator, and the app **falls back to CPU automatically**
if the GPU can't be acquired or errors mid-run.

From this folder, with the HF CLI authenticated (`hf auth login`):

```bash
pip install gradio
gradio deploy
```

Then set the Space hardware to **ZeroGPU** (Settings → Hardware; requires a PRO
account). `gradio deploy` uploads the directory and runs `app.py` against
`requirements.txt`.

Notes:
- **torch is pinned to 2.8.0** — ZeroGPU requires one of 2.8.0 / 2.9.1 / 2.10.0 /
  2.11.0; PyG 2.6.1 and Lightning 2.6.5 pair with it. `torch-scatter`/`-sparse`
  and `torchvision` are not needed.
- The checkpoint is a **slim, inference-only** re-save of `last-v1.ckpt`
  (`state_dict` + cfg, ~3 MB). The original embedded Lightning-2.0 Trainer state,
  which won't unpickle under Lightning 2.6. To regenerate it from a full
  checkpoint, see `last-v1.ckpt.orig` handling in the commit history.
- A generation run must finish inside ZeroGPU's per-call window (the decorator
  requests up to 120 s), so keep the target phrase count modest.

**Local / CPU / standard-GPU hosts.** `spaces` is optional — without it the
`@spaces.GPU` decorator is a no-op, so the app runs in-process and uses CUDA when
present, else CPU. A [`Dockerfile`](Dockerfile) is included for self-hosting
(`docker run --gpus all -p 7860:7860 progress-ui`; drop `--gpus all` for CPU).

## Workflow (three tabs)

### Tab 1 · Browse & Select Melodies
1. Click **Load Phrase Library** – loads all pre-generated phrases from
   `ProGress_Supplement/phrase_stitching/diffusion_output/` and applies
   rejection sampling (illegal harmonics, bad mode mixture, bad counterpoint).
   Results are cached in `.phrase_cache.json` for fast re-loads.
2. Filter by mode (major / minor / mixed) and start/end harmony.
3. Enter a phrase **ID**, click **Preview** to hear it in the embedded MIDI player.
4. Click **Mark as Favourite** to tag phrases you like –
   the stitcher will preferentially draw from your selection.

### Tab 2 · Generate New Phrases  *(requires SchenkerDiff checkpoint)*
- Place `last-v1.ckpt` in `SchenkerDiff/` to enable this tab.
- Adjust batch size and click **Generate** – the SchenkerDiff diffusion model
  runs 100 denoising steps and realises each output graph as a 2-voice score.
- Phrases that pass the same rejection filters are offered for addition to the pool.

### Tab 3 · Stitch & Export
1. Choose a **harmonic structure** (e.g., I – V – I, i – III – iv – i …).
2. Click **Stitch!** – four phrase sections are sampled (preferring favourites),
   transposed to match the progression, inner voices are filled, and the
   sections are concatenated into a full piece.
3. Listen to each section and the full composition in the embedded MIDI player.
4. Click **Download MIDI** to save the result.
5. **Resample** to try a different combination with the same structure.

## Package layout

Everything below ships inside this folder — no external directories needed.

```
ProGress_UI/                    ← deployable package (this folder)
├── app.py                      ← Gradio entrypoint (HF Space app_file)
├── backend.py                  ← API layer; resolves paths to ./vendor
├── requirements.txt            ← pinned CPU runtime
├── Dockerfile / .dockerignore  ← portable container build
├── .gitattributes              ← Git LFS for *.ckpt, *.pt
└── vendor/                     ← vendored runtime dependencies
    ├── phrase_stitching/       ← from ProGress_Supplement
    │   ├── *.py                ← stitching / rejection-sampling logic
    │   └── diffusion_output/   ← phrase library (output_graphs_{1-13}/*.xml)
    └── SchenkerDiff/           ← trimmed model code (no training data/checkpoints)
        ├── last-v1.ckpt        ← diffusion checkpoint (LFS) → enables Tab 2
        ├── inference.py, src/, output_vis/, configs/
        └── data/schenker/…/processed/*.pt   ← conditioning tensors (LFS)
```

The vendored copy excludes everything not used at inference time (training
checkpoints, `saved_models/`, the `schenkerian_clusters/` dataset, `wandb/`,
visualization HTML, notebooks), bringing the SchenkerDiff dependency from
~858 MB down to ~78 MB.

## Notes

- The MIDI player uses [html-midi-player](https://github.com/cifkao/html-midi-player)
  (loaded from CDN) with Magenta sound fonts.  Internet access is required for audio playback.
- First-time phrase loading takes **~20 s** on this machine for the full 1 200-file library
  (about **90 phrases** typically pass the rejection filters).  Subsequent loads use
  the JSON cache and take **~2 s**.

## Smoke-test status

End-to-end smoke tests inside the `digress` conda env (`/home/peter/miniconda3/envs/digress`):

| Path | Status | Notes |
|------|--------|-------|
| Phrase load + rejection sampling   | ✅ works | 90/1200 valid, 20 s |
| Cache reload                       | ✅ works | 1.5 s |
| MIDI byte conversion               | ✅ works | valid MThd header |
| All 5 stitch structures            | ✅ works | each produces a 4-part / 8-measure MIDI |
| Favourite-preference stitching     | ✅ works | favourites picked when compatible |
| Gradio app build                   | ✅ works | 64 blocks, no errors |
| **SchenkerDiff `generate_new_phrases()`** | ✅ works | batch=2 → ~60 s, batch=4 → ~50 s on CPU; ~50 % rejection-pass rate |

### SchenkerDiff generation (Tab 2)

`backend.generate_new_phrases()` runs the full diffusion model end-to-end:
load checkpoint → sample conditioning E/r from a processed `.pt` file → 100
DDIM steps → `realization.py` → rejection-sampling filters.

Several upstream issues in the SchenkerDiff repo had to be worked around inside
`backend.generate_new_phrases()`:

1. **`graph_tool` import in `src/analysis/spectre_utils.py`** fails on this env
   with a `libgomp` symbol mismatch.  Stubbed in `sys.modules` (the only consumer
   of that module is the training-time sampling metric, which we never call).
2. **`PlanarSamplingMetrics`** is constructed inside `initialize_model()` and is
   pickled as part of the checkpoint's module tree, so the stub must subclass
   `torch.nn.Module` for `named_modules()` to work during deserialisation.
3. **Checkpoint was saved on CUDA**.  PyTorch Lightning explicitly passes
   `map_location=None`, so `torch.load` is monkey-patched to default to CPU when
   no GPU is available.
4. **`inference.sample_r_E()` hardcodes `E_sample` to shape `(m, m, 10)`** but the
   model's `Edim_output` is 30 and the regenerated `.pt` files have edge_attr
   width 30.  Reimplemented in `backend.py` so it reads the dimension from
   `model.limit_dist.E`.

### Regenerating processed data

If `SchenkerDiff/data/schenker/processed/heterdatacleaned/processed/` is empty
(or out of date with the dataset code), run:

```bash
python ProGress_UI/regenerate_processed.py
```

This invokes `SchenkerGraphDataModule(cfg)`, which triggers PyG's
`Dataset._process()` → `process_file()` for every XML in `train-names.txt`,
writing `0_processed.pt … N_processed.pt` into the processed dir.  Takes
~4 min for the full 1 780-file run on this machine.  No training, no GPU,
no checkpoint writes.
