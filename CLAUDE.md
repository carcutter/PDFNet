# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PDFNet — PyTorch implementation of "High-Precision Dichotomous Image Segmentation via Depth Integrity-Prior and Fine-Grained Patch Strategy" (CVPR 2026). The core idea is that pseudo-depth maps of foreground objects have *lower variance* than background, and the model exploits this "depth integrity prior" as a regularizer alongside RGB segmentation.

## Environment

Conda env per the README:

```bash
conda create -n PDFNet python=3.11.4
conda activate PDFNet
pip install -r requirements.txt
```

Note: `requirements.txt` is lightweight (no version pins on opencv-python, gradio, etc.) and `metric_tools/Test.py` additionally imports `ttach` which is not in `requirements.txt` — install it explicitly if running evaluation.

## Common commands

| Task | Command |
|------|---------|
| Train from scratch (DIS-5K) | `python Train_PDFNet.py` |
| **Build dataset index CSV (run once)** | `python build_dataset_csv.py` (writes `./data/index.csv` from interior-segmentation subdirs) |
| **Fine-tune (full) — CSV + 80/20 split** | `python Finetune_PDFNet.py --checkpoint checkpoints/PDFNet_Best.pth --synthesize_depth` |
| **Fine-tune (LoRA) — CSV + 80/20 split** | `python Finetune_PDFNet.py --mode lora --checkpoint checkpoints/PDFNet_Best.pth --lora_rank 8 --lora_alpha 16 --synthesize_depth` |
| Download pretrained weights | `python download_checkpoint.py` (writes Swin-B backbone + PDFNet checkpoint to `./checkpoints/`) |
| Quick demo / inference | open `demo.ipynb` |
| Evaluation on DIS test set | `cd metric_tools && python Test.py` |
| Generate pseudo-depth maps | run `DAM_V2/Depth-prepare.ipynb` |
| Resume / legacy finetune | `python Train_PDFNet.py --finetune <path.pth> --finetune_epoch <N>` |
| Debug run | add `--DEBUG True` (suppresses checkpoint + tensorboard writes) |
| Switch eval metric | `--eval_metric F1` (default) or `--eval_metric MAE` |
| TensorBoard | `tensorboard --logdir runs/` |

There is no test suite, linter, or formatter configured.

## Fine-tuning (Finetune_PDFNet.py)

`Finetune_PDFNet.py` + `finetune_main.py` is a self-contained fine-tuning entry point. It's parallel to `Train_PDFNet.py`/`main.py` and shares the model code in `models/PDFNet.py`.

Differences from `Train_PDFNet.py`:

- **Loads a pretrained checkpoint** (`--checkpoint`, default `checkpoints/PDFNet_Best.pth`) with `strict=False` so it tolerates LoRA-injected models or partial loads.
- **Two training modes** (`--mode`):
  - `full` — train all parameters. Add `--freeze_backbone` to freeze the Swin-B encoder.
  - `lora` — freeze the base model and inject LoRA adapters into matching modules (`models/lora.py`). Default targets are `decoder.*` and `depth_decoder.*`. The MHA out_proj inside `CoA` is *not* wrapped (PyTorch's `nn.MultiheadAttention.forward` accesses `self.out_proj.weight` directly and would break).
- **CSV-driven dataset (`--dataset_builder csv`, default)** — used for this repo's interior-segmentation data. Build the index once:
  ```bash
  python build_dataset_csv.py            # scans ./data/*interior_segmentation*/{raw,raw_images}+masks
                                         # writes ./data/index.csv
  ```
  The CSV has columns `image_path, mask_path, depth_path, dataset` (paths relative to `--data_path`, default `./data`). The fine-tune script splits it 80/20 with `--val_split 0.2 --csv_split_seed 42`. The same seed gives the same split across runs. Depth maps are not present yet; pass `--synthesize_depth` to fall back to RGB grayscale until DAM-V2 is run.
- **`./data/{train,val}` layout** via `--dataset_builder finetune`:
  ```
  ./data/
    train/{images,masks,depth}/...
    val/{images,masks,depth}/...
  ```
  Per-variant depth dirs (`depth_small`, `depth_base`, `depth_large`, `depth_large_1024`) are auto-detected; missing ones fall back to `depth/`. Set `--dataset_builder dis` to use the original DIS layout instead.
- **TensorBoard logging** (under `runs/<model>_<mode>_<timestamp>/`):
  - `Inputs/augmented` — first training batch each epoch, shown as `[image | GT | depth | overlay]` per sample with file names + label indices in a sibling `_meta` text panel. Use this to audit augmentations.
  - `Val/inputs`, `Val/predictions` — one validation batch + the model's output each epoch.
  - `Train/loss`, `Train/seg_loss`, `Val/loss`, `Val/seg_loss`, `Val/F1`, `Val/MAE`, `Val/IoU`, `Optim/lr`.

LoRA checkpoint format: `{"mode": "lora", "lora": <state_dict of only lora_A/lora_B params>}`. To reload: rebuild the model, `load_pretrained(model, "checkpoints/PDFNet_Best.pth")`, call `inject_lora(...)` with the same rank/targets, then load the LoRA state dict.

### Pre-existing bug fixed for fine-tuning

`models/PDFNet.py:289` had `# self.depth_decoder = depth_decoder` commented out — but the attribute is referenced in `forward()` at line ~450. As shipped, `Train_PDFNet.py`'s first forward pass would crash with `AttributeError`. The fine-tuning prep restores that assignment so both training entry points actually run.

## Required external assets

Training will fail without these — they are **not** in the repo:

- `DATA/DIS-DATA/{DIS-TR,DIS-VD,DIS-TE1..4}/{images,masks,depth_small,depth_base,depth_large,depth_large_1024}/` — the DIS-5K dataset *plus four pre-generated depth variants per split* (see "Depth data layout" below).
- `checkpoints/swin_base_patch4_window12_384_22k.pth` — Swin-B backbone weights (loaded by `SwinB(pretrained=True)` in `models/PDFNet.py`).
- `checkpoints/PDFNet_Best.pth` — released weights, used by `metric_tools/Test.py`.
- `DAM_V2/` — clone of Depth-Anything-V2 repo, used only to generate depth maps once.

## Architecture

The pipeline is end-to-end and **always uses both RGB and pseudo-depth as inputs** (also at inference). It cannot run on RGB alone.

```
RGB + pseudo-depth ─┐
                    ├─► encoder (Swin-B, shared)
                    │     ├─ downsampled image stream (RGB)
                    │     ├─ downsampled depth stream (depth as 3-ch)
                    │     └─ patched image stream (image split into patch_ratio² tiles)
                    │
                    ├─► PDF_decoder
                    │     └─ FSE modules at L4→L1, each running CoA cross-attention
                    │        between {img, depth, patch} features, guided by BIS
                    │        (Boundary-aware Integrity Selection from previous side-pred)
                    │
                    └─► PDF_depth_decoder (auxiliary depth head, supervised by SiLog)

Losses (summed in PDFNet_process.forward):
  segmentation:  structure_loss + 0.5·SSIM    over [final, side_1..side_4]
  integrity:     IntegrityPriorLoss(pred·depth, gt)   ×  1/2
  depth aux:     SiLogLoss(pred_depth, depth_gt)      ×  1/10
```

Three things to know before editing the model:

1. **Three encoder passes per step.** `PDFNet_process.encode` is called *three times* per forward: once on the downsampled image+depth (concatenated along batch dim, split apart later), and once on the patched image batch (`B × patch_ratio²` items). Memory and time scale with `patch_ratio` (default 8 → 64 patches per image). This is the dominant cost.

2. **`build_model` returns `(model, model_name)`**, not just the model — a tuple. The model name is used throughout `main.py` to construct checkpoint/tensorboard/valid_sample directory names.

3. **Decoder coupling.** `PDFNet_process.decoder` expects three parallel feature pyramids (img, depth, patch) with matching channel widths after the `channel_mix{1..4}` projections. The depth decoder consumes the *concatenation* of the three. Changing emb dim (`--emb`, default 128) ripples through `make_crs` calls inside `PDF_decoder`/`PDF_depth_decoder`.

## Dataloader contract

`dataloaders/Mydataset.py`'s `MyDataset.__getitem__` returns a dict with keys `image, gt, depth, depth_large, label, image_name, image_size`. The training loop **requires all of these** — `main.py:135` unpacks `data['image'], data['gt'], data['label'], data['depth'], data['depth_large']` and the model signature is `forward(img, depth, gt, depth_gt)`.

### Depth data layout

For each image at `<split>/images/<name>.jpg`, the dataloader expects:

- `<split>/depth_small/<name>.jpg`
- `<split>/depth_base/<name>.jpg`
- `<split>/depth_large/<name>.jpg`
- `<split>/depth_large_1024/<name>.jpg` (training only — read as `depth_large` and used as depth GT for the SiLog auxiliary loss)

During training, one of `{depth_small, depth_base, depth_large}` is sampled randomly per item (lines 388–393) — this is depth augmentation. During validation only `depth_large` is used. If you only have one depth variant, point all three paths at the same files, or modify the random branch.

### Label parsing

`label` is a one-hot vector indexed by the **first three `#`-separated tokens of the filename** (e.g. `5#Artifact#1#Basket#...jpg` → key `5Artifact1`). Custom datasets without this naming convention will produce `KeyError`s unless you pass `use_gt=False` (which the dataloader supports) or rewrite the stoi logic in `MyDataset.__init__`.

## Training-loop quirks

- **Hard-example mining toggle** (`--update_half True`): on odd epochs, only re-trains on the half of samples with the highest loss from the previous epoch. Off by default.
- **Gradient accumulation:** `--update_freq N` accumulates N batches before stepping. With `--batch_size 1` (default) this is the practical lever for effective batch size — bumping `--batch_size` is VRAM-bound at 1024² input.
- **Mixed precision** is always on (`torch.cuda.amp.autocast` + `GradScaler`); no flag to disable.
- **Project snapshotting:** with `--COPY True` (default) every run copies the working tree into `valid_sample/<model><timestamp>/project_copy/`. This can eat disk fast; disable for quick experiments.
- **Checkpoint retention:** `keep_n_files(this_checkpoints_dir, n=3)` keeps only the 3 most recent best checkpoints per run.

## Evaluation script caveats

`metric_tools/Test.py` is **partially broken as-shipped** — treat it as a template, not a working script:

- Line 21: `from Train_VIDIF import get_args_parser` — should be `from Train_PDFNet import get_args_parser`.
- Line 36, 45–49, 56: hardcoded absolute paths under `/home/PDFNet/...` and `/home/DATA/...` that must be edited for your environment.
- Imports `ttach` (TTA library) which is not in `requirements.txt`.
- Calls `soc_metrics(file_name)` from `metric_tools/soc_metrics.py`; that module's `gt_roots`/`cycle_roots` likely also need editing.

When asked to "run evaluation," surface these issues rather than silently running a broken script.

## Conventions worth preserving

- The codebase mixes English and Chinese comments (the author's first language). Don't translate existing comments unless asked.
- Many helpers are duplicated between `Train_PDFNet.py` and `main.py` (e.g. `get_files`, `get_args_parser`). `args.py` also exists as a third copy of the arg parser — `Train_PDFNet.py` uses its *own* inline parser, not `args.py`. If editing CLI args, update `Train_PDFNet.py:get_args_parser` (the one actually used), and consider whether `args.py` should be kept in sync.
- `models/utils.py` (loss/op helpers) and `models/util.py` both exist — they are different files; check which is imported.
