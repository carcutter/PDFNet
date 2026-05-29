"""Run PDFNet inference and save inputs + predicted masks.

Supports two kinds of fine-tuned checkpoints (selected by --mode):

  - lora: builds the base model, loads --checkpoint, injects LoRA adapters with
          the same rank/alpha/targets, then strict-loads the LoRA-only state
          dict ({"mode": "lora", "lora": ...}) from --finetune_checkpoint.
  - full: builds the base model and loads --finetune_checkpoint as a full
          state dict (shape-filtered, strict=False).

Inputs come from either:
  - the validation split of the training CSV (default; same seed gives the same
    split as Finetune_PDFNet.py), or
  - every image under --image_dir.

Postprocessing mirrors metric_tools/Test.py:
  - ttach TTA: HorizontalFlip x Scale([0.75, 1.0, 1.25])
  - per-transform `model.inference`, deaugment, mean across TTA, sigmoid
  - PIL bilinear resize back to the original image size
  - optional binarisation at --threshold (default 0.5)

Outputs (default `results/`):
  <output_dir>/<stem>_input.png   - denormalized RGB at original resolution
  <output_dir>/<stem>_mask.png    - predicted mask at original resolution

Quickstart:

  # LoRA fine-tune (mode auto-picked from config/training/lora.yaml)
  python run_inference.py \\
      --finetune_checkpoint checkpoints/finetune/PDFNet_swinB_lora_.../LAST.pth

  # Full-decoder fine-tune
  python run_inference.py --config config/training/full_decoder.yaml \\
      --finetune_checkpoint checkpoints/finetune/PDFNet_swinB_full_.../LAST.pth

  # On a folder of images (no GT/masks)
  python run_inference.py \\
      --finetune_checkpoint .../LAST.pth \\
      --image_dir path/to/images --synthesize_depth
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import ttach as tta
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.Mydataset import (
    GOSNormalize,
    MyDataset,
    _load_pair_list_from_csv,
    _split_pairs,
)
from models import lora as lora_lib
from models.PDFNet import build_model
from finetune_main import load_pretrained


def get_args_parser():
    p = argparse.ArgumentParser("PDFNet LoRA test", add_help=False)
    p.add_argument("--config", default="config/training/lora.yaml", type=str,
                   help="YAML providing defaults (same file used at fine-tuning time so "
                        "the LoRA rank/alpha/targets line up). Pass --config '' to disable.")

    # ---- model ----
    p.add_argument("--model", default="PDFNet_swinB", type=str)
    p.add_argument("--back_bone", default="PDFNet_swinB", type=str)
    p.add_argument("--back_bone_channels_stage1", default=128, type=int)
    p.add_argument("--back_bone_channels_stage2", default=256, type=int)
    p.add_argument("--back_bone_channels_stage3", default=512, type=int)
    p.add_argument("--back_bone_channels_stage4", default=1024, type=int)
    p.add_argument("--emb", default=128, type=int)
    p.add_argument("--input_size", default=1024, type=int)
    p.add_argument("--Crop_size", default=1024, type=int)
    p.add_argument("--drop_path", default=0.1, type=float)

    # ---- checkpoints ----
    p.add_argument("--checkpoint", default="checkpoints/PDFNet_Best.pth", type=str,
                   help="base PDFNet checkpoint (loaded with strict=False). "
                        "Always loaded first; the fine-tuned weights are applied on top.")
    p.add_argument("--finetune_checkpoint", required=True, type=str,
                   help="fine-tuned checkpoint to load on top of --checkpoint. "
                        "Interpreted per --mode: lora -> {'mode':'lora','lora':...}; "
                        "full -> a plain model.state_dict().")
    p.add_argument("--mode", default="lora", choices=["lora", "full"],
                   help="how to load --finetune_checkpoint. Picked up from the YAML "
                        "config (lora.yaml -> lora; full_decoder.yaml -> full).")
    p.add_argument("--lora_rank", default=8, type=int)
    p.add_argument("--lora_alpha", default=16.0, type=float)
    p.add_argument("--lora_dropout", default=0.0, type=float)
    p.add_argument("--lora_targets", nargs="*", default=None,
                   help="fnmatch patterns for module names to wrap. Defaults to "
                        "lora_lib.DEFAULT_DECODER_TARGETS.")

    # ---- input source ----
    p.add_argument("--image_dir", default="", type=str,
                   help="if set, run on every image under this directory (no GT/masks). "
                        "Otherwise the val split of --csv_path is used.")
    p.add_argument("--data_path", default="./data/interior_segmentation", type=str)
    p.add_argument("--csv_path", default="data/interior_segmentation/index.csv", type=str)
    p.add_argument("--val_split", default=0.2, type=float)
    p.add_argument("--csv_split_seed", default=42, type=int)
    p.add_argument("--synthesize_depth", action="store_true",
                   help="fall back to RGB grayscale when a depth file isn't found")
    p.add_argument("--mask_mode", default="red_green", choices=["red_green", "grayscale"])
    p.add_argument("--depth_variants", nargs="*",
                   default=["depth_large", "depth_base", "depth_small"])
    p.add_argument("--depth_gt_dir", default="depth_large_1024", type=str)
    p.add_argument("--depth_fallback_dir", default="depth", type=str)

    # ---- runtime ----
    p.add_argument("--output_dir", default="results", type=str,
                   help="root results dir; outputs go under <output_dir>/<exp_name>/")
    p.add_argument("--exp_name", default="", type=str,
                   help="experiment subfolder under --output_dir. Defaults to the "
                        "name of the directory containing --finetune_checkpoint "
                        "(e.g. PDFNet_swinB_lora_2026-05-25_15_05_04).")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--batch_size", default=1, type=int,
                   help="batch size; >1 only works when all inputs are the same size")
    p.add_argument("--limit", default=0, type=int,
                   help="if >0, stop after this many samples (useful for smoke runs)")
    p.add_argument("--threshold", default=0.5, type=float,
                   help="binarisation threshold applied to the sigmoid probability "
                        "map before saving (pixels >= threshold -> 255, else 0). "
                        "Set <= 0 to save the raw grayscale probability instead.")
    p.add_argument("--chached", default=False, type=bool)
    return p


def _apply_yaml_defaults(parser: argparse.ArgumentParser, config_path: str) -> None:
    if not config_path:
        return
    path = Path(config_path)
    if not path.is_file():
        print(f"[config] {path} not found - using built-in defaults", flush=True)
        return
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    valid = {a.dest for a in parser._actions if a.dest != "help"}
    # Silently drop unknown training-only keys (e.g. opt, epochs) so the same
    # YAML used for fine-tuning is reusable here.
    data = {k: v for k, v in data.items() if k in valid}
    parser.set_defaults(**data)
    print(f"[config] loaded {len(data)} keys from {path}", flush=True)


def _gather_image_pairs(image_dir: str):
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    pairs = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(exts):
                pairs.append({"image": os.path.join(root, f), "mask": None, "depth": None})
    pairs.sort(key=lambda p: p["image"])
    return pairs


def _build_dataset(args):
    """Either val split of the CSV (default) or every image in --image_dir."""
    transform = [GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    if args.image_dir:
        pairs = _gather_image_pairs(args.image_dir)
        if not pairs:
            raise FileNotFoundError(f"No images under {args.image_dir}")
        print(f"[data] image_dir={args.image_dir}: {len(pairs)} images")
        return MyDataset(
            pair_list=pairs,
            transform=transform,
            chached=False,
            size=[args.input_size, args.input_size],
            istrain=False,
            use_gt=False,
            depth_variants=tuple(args.depth_variants),
            depth_gt_dir=args.depth_gt_dir,
            depth_fallback_dir=args.depth_fallback_dir,
            labels_from_filename=False,
            synthesize_missing_depth=args.synthesize_depth,
            mask_mode=args.mask_mode,
        )

    pairs = _load_pair_list_from_csv(args.csv_path, args.data_path)
    _, val_pairs = _split_pairs(pairs, args.val_split, args.csv_split_seed)
    print(f"[data] {args.csv_path}: {len(pairs)} total -> val={len(val_pairs)} "
          f"(seed={args.csv_split_seed}, val_frac={args.val_split})")
    return MyDataset(
        pair_list=val_pairs,
        transform=transform,
        chached=False,
        size=[args.input_size, args.input_size],
        istrain=False,
        use_gt=True,
        depth_variants=tuple(args.depth_variants),
        depth_gt_dir=args.depth_gt_dir,
        depth_fallback_dir=args.depth_fallback_dir,
        labels_from_filename=False,
        synthesize_missing_depth=args.synthesize_depth,
        mask_mode=args.mask_mode,
    )


def _load_lora_ckpt(model, ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not (isinstance(ckpt, dict) and ckpt.get("mode") == "lora" and "lora" in ckpt):
        raise ValueError(
            f"{ckpt_path} is not a LoRA checkpoint "
            "(expected {'mode': 'lora', 'lora': <state_dict>}). "
            "If this is a full-mode fine-tune, pass --mode full."
        )
    lora_lib.load_lora_state_dict(model, ckpt["lora"], strict=True)
    return len(ckpt["lora"])


def _load_full_ckpt(model, ckpt_path: str) -> int:
    """Load a full-decoder fine-tune checkpoint (plain model.state_dict)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and ckpt.get("mode") == "lora":
        raise ValueError(
            f"{ckpt_path} is a LoRA-only checkpoint but --mode is 'full'. "
            "Pass --mode lora to load it."
        )
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape}
    missing = set(model_state) - set(filtered)
    unexpected = set(state) - set(filtered)
    model.load_state_dict(filtered, strict=False)
    print(f"[full] loaded {len(filtered)}/{len(model_state)} tensors from {ckpt_path}; "
          f"skipped {len(unexpected)} (key/shape mismatch); "
          f"{len(missing)} model keys left at base init")
    return len(filtered)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denormalize_to_pil(image_t: torch.Tensor) -> Image.Image:
    """ImageNet-denormalize a (3,H,W) tensor and return a PIL RGB image."""
    img = image_t.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("PDFNet LoRA test", parents=[get_args_parser()])
    prelim, _ = parser.parse_known_args()
    _apply_yaml_defaults(parser, prelim.config)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    exp_name = args.exp_name or Path(args.finetune_checkpoint).parent.name
    out_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[output] saving to {out_dir}/")

    print(f"[model] building {args.model} ...")
    model, _ = build_model(args)
    load_pretrained(model, args.checkpoint)

    if args.mode == "lora":
        targets = list(args.lora_targets) if args.lora_targets else list(lora_lib.DEFAULT_DECODER_TARGETS)
        wrapped = lora_lib.inject_lora(
            model,
            targets=targets,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        if not wrapped:
            raise RuntimeError(f"No modules matched LoRA targets {targets}.")
        n_loaded = _load_lora_ckpt(model, args.finetune_checkpoint)
        print(f"[lora] wrapped {len(wrapped)} modules, loaded {n_loaded} LoRA tensors "
              f"from {args.finetune_checkpoint}")
    elif args.mode == "full":
        _load_full_ckpt(model, args.finetune_checkpoint)
    else:
        raise ValueError(f"Unknown --mode {args.mode}")

    model.to(device).eval()

    ds = _build_dataset(args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)

    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1.0, 1.25], interpolation="bilinear", align_corners=False),
    ])
    to_pil = T.ToPILImage()

    pbar = tqdm(loader, desc="infer", total=len(loader))
    n_saved = 0
    for data in pbar:
        if args.limit and n_saved >= args.limit:
            break
        inputs = data["image"].to(device, non_blocking=True)
        depth = data["depth"].to(device, non_blocking=True)
        names = data["image_name"]
        # image_size: list of two stacked tensors [H, W] after DataLoader collation.
        h_t, w_t = data["image_size"]

        masks = []
        for transformer in tta_transforms:
            rgb_trans = transformer.augment_image(inputs)
            depth_trans = transformer.augment_image(depth)
            _, pred_logits = model.inference(rgb_trans, depth_trans)
            deaug = transformer.deaugment_mask(pred_logits)
            masks.append(deaug)
        prediction = torch.mean(torch.stack(masks, dim=0), dim=0).sigmoid()

        for k in range(inputs.shape[0]):
            stem = Path(str(names[k])).stem
            orig_h = int(h_t[k].item() if torch.is_tensor(h_t) else h_t[k])
            orig_w = int(w_t[k].item() if torch.is_tensor(w_t) else w_t[k])

            # Resize the probability map first (bilinear keeps boundaries
            # smooth), then binarise at the threshold so we save a clean 0/255
            # PNG instead of an anti-aliased one.
            mask_pil = to_pil(prediction[k].cpu()).resize((orig_w, orig_h), Image.BILINEAR)
            if args.threshold > 0:
                mask_arr = np.array(mask_pil)
                mask_arr = (mask_arr >= int(round(args.threshold * 255))).astype(np.uint8) * 255
                mask_pil = Image.fromarray(mask_arr, mode="L")
            mask_pil.save(os.path.join(out_dir, f"{stem}_mask.png"))

            img_pil = _denormalize_to_pil(inputs[k])
            img_pil = img_pil.resize((orig_w, orig_h), Image.BILINEAR)
            img_pil.save(os.path.join(out_dir, f"{stem}_input.png"))

            n_saved += 1
            if args.limit and n_saved >= args.limit:
                break

    pbar.close()
    print(f"[done] saved {n_saved} input+mask pairs to {out_dir}/")


if __name__ == "__main__":
    main()
