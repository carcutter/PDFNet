"""Fine-tuning loop for PDFNet.

Differences from main.py:
- Loads a pretrained PDFNet checkpoint and supports either full fine-tune or
  LoRA-only training (--mode {full,lora}). LoRA freezes the base model and
  injects low-rank adapters into decoder.* and depth_decoder.*.
- Logs the augmented training batch (image / GT / depth / overlay) to
  TensorBoard so dataloader augmentations can be visually audited.
- Logs both training and validation losses (total, segmentation-only) plus
  F1/MAE/IoU on the validation set every epoch.
"""

from __future__ import annotations

import datetime
import gc
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from timm.scheduler import create_scheduler
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utiles
from dataloaders import Mydataset
from metric_tools.F1torch import f1score_torch
from models.PDFNet import build_model
from models import lora as lora_lib


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def copy_allfiles(src, dest, not_case=("valid_sample", "runs")):
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        if any(skip in rel for skip in not_case):
            continue
        out = os.path.join(dest, rel)
        os.makedirs(out, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(root, f), os.path.join(out, f))


def denormalize_image(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display. x: (B,3,H,W) or (3,H,W)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    mean = IMAGENET_MEAN.to(x.device)
    std = IMAGENET_STD.to(x.device)
    return (x * std + mean).clamp(0.0, 1.0)


def make_overlay(image: torch.Tensor, mask: torch.Tensor, color=(1.0, 0.2, 0.2), alpha=0.5):
    """Blend a binary mask onto an image. image: (B,3,H,W) in [0,1], mask: (B,1,H,W)."""
    mask = mask.clamp(0.0, 1.0)
    color_t = torch.tensor(color, device=image.device).view(1, 3, 1, 1)
    overlay = image * (1 - mask * alpha) + color_t * (mask * alpha)
    return overlay.clamp(0.0, 1.0)


def to_3ch(x: torch.Tensor) -> torch.Tensor:
    """Expand a single-channel tensor to 3 channels along dim=1."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def log_augmented_batch(writer: SummaryWriter, batch: dict, step: int,
                        tag: str = "Inputs/augmented", n: int = 4) -> None:
    """Write a TensorBoard image grid summarising the augmented batch.

    Per sample, four rows are shown side-by-side: input RGB (denormalized),
    GT mask, depth, and the GT-mask overlay on the RGB image. File names and
    label indices are logged as a sibling text panel.
    """
    image = batch["image"][:n].detach().cpu().float()
    gt = batch["gt"][:n].detach().cpu().float()
    depth = batch["depth"][:n].detach().cpu().float()

    image_rgb = denormalize_image(image)
    gt_3 = to_3ch(gt)
    depth_3 = to_3ch(depth)
    overlay = make_overlay(image_rgb, gt)

    # Stack [image, gt, depth, overlay] horizontally per sample
    panel = torch.cat([image_rgb, gt_3, depth_3, overlay], dim=3)
    grid = vutils.make_grid(panel, nrow=1, padding=4, pad_value=1.0)
    writer.add_image(tag, grid, global_step=step)

    # Sidecar text: filenames + label indices
    names = batch.get("image_name", [])[:n]
    labels = batch.get("label", torch.empty(0))
    if isinstance(labels, torch.Tensor) and labels.numel() > 0:
        label_idx = labels[:n].argmax(dim=-1).tolist()
    else:
        label_idx = ["-"] * len(names)
    lines = [
        f"| # | label | filename |",
        f"|---|-------|----------|",
    ]
    for i, (name, li) in enumerate(zip(names, label_idx)):
        lines.append(f"| {i} | {li} | `{Path(str(name)).name}` |")
    writer.add_text(f"{tag}_meta", "\n".join(lines), global_step=step)


# ---------------------------------------------------------------------------
# Checkpoint loading & LoRA setup
# ---------------------------------------------------------------------------

def load_pretrained(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load a PDFNet checkpoint, skipping keys whose shape changed or are missing."""
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape}
    missing = set(model_state) - set(filtered)
    unexpected = set(state) - set(filtered)
    model.load_state_dict(filtered, strict=False)
    print(f"[checkpoint] loaded {len(filtered)}/{len(model_state)} tensors from {ckpt_path}")
    if missing:
        print(f"[checkpoint] missing in ckpt: {len(missing)} keys "
              f"(first: {sorted(missing)[0] if missing else '-'})")
    if unexpected:
        print(f"[checkpoint] unexpected in ckpt (skipped): {len(unexpected)} keys "
              f"(first: {sorted(unexpected)[0] if unexpected else '-'})")


def setup_finetune_mode(model: torch.nn.Module, args) -> dict:
    """Apply --mode-specific freezing / LoRA injection. Returns a meta dict."""
    meta = {"mode": args.mode}
    if args.mode == "lora":
        targets = list(args.lora_targets) if args.lora_targets else list(lora_lib.DEFAULT_DECODER_TARGETS)
        wrapped = lora_lib.inject_lora(
            model,
            targets=targets,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        lora_lib.mark_only_lora_as_trainable(model, bias="none")
        meta["lora_wrapped_count"] = len(wrapped)
        meta["lora_wrapped_first"] = wrapped[:5]
        if not wrapped:
            raise RuntimeError(
                f"No modules matched LoRA targets {targets}. Check --lora_targets."
            )
    elif args.mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        if args.freeze_backbone:
            for p in model.encoder.parameters():
                p.requires_grad = False
            meta["backbone_frozen"] = True
    else:
        raise ValueError(f"Unknown --mode {args.mode}")
    trainable, total = lora_lib.count_trainable(model)
    meta["trainable_params"] = trainable
    meta["total_params"] = total
    meta["trainable_frac"] = trainable / max(total, 1)
    return meta


def save_checkpoint(model: torch.nn.Module, path: str, args) -> None:
    if args.mode == "lora":
        torch.save({"mode": "lora", "lora": lora_lib.lora_state_dict(model)}, path)
    else:
        torch.save(model.state_dict(), path)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def iou_from_pred(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    p = (pred > thr).float()
    g = (gt > 0.5).float()
    inter = (p * g).sum().item()
    union = (p + g - p * g).sum().item()
    return (inter + eps) / (union + eps)


# ---------------------------------------------------------------------------
# Training / validation epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, args, epoch, writer):
    model.train()
    pbar = tqdm(loader, desc=f"train {epoch+1}/{args.epochs}")
    sums = {"total": 0.0, "target": 0.0}
    n_iters = 0
    logged_aug_this_epoch = False

    for it, data in enumerate(pbar):
        if args.smoke_test and it >= args.smoke_test:
            break
        if writer is not None and not logged_aug_this_epoch and (
            epoch % max(args.log_batch_every, 1) == 0
        ):
            log_augmented_batch(writer, data, step=epoch, n=args.log_grid_n)
            logged_aug_this_epoch = True

        inputs, gt = data["image"], data["gt"]
        depth, depth_large = data["depth"], data["depth_large"]
        if args.device != "cpu":
            inputs = inputs.to(args.device, non_blocking=True)
            gt = gt.to(args.device, non_blocking=True)
            depth = depth.to(args.device, non_blocking=True)
            depth_large = depth_large.to(args.device, non_blocking=True)

        with autocast():
            _, loss, target_loss = model(
                Variable(inputs, requires_grad=False),
                Variable(depth, requires_grad=False),
                Variable(gt, requires_grad=False),
                Variable(depth_large, requires_grad=False),
            )
            loss_scaled = loss / args.update_freq

        scaler.scale(loss_scaled).backward()
        if (it + 1) % args.update_freq == 0 or (it + 1) == len(loader):
            if args.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.clip_grad,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        sums["total"] += float(loss.detach().cpu().item())
        sums["target"] += float(target_loss.detach().cpu().item())
        n_iters += 1
        pbar.set_postfix(loss=f"{sums['total']/n_iters:.4f}",
                         seg=f"{sums['target']/n_iters:.4f}")

        del loss, loss_scaled, target_loss
        gc.collect()

    pbar.close()
    return {
        "loss": sums["total"] / max(n_iters, 1),
        "seg_loss": sums["target"] / max(n_iters, 1),
    }


@torch.no_grad()
def validate(model, loader, args, epoch, writer, save_dir=None):
    model.eval()
    pbar = tqdm(loader, desc=f"val   {epoch+1}/{args.epochs}")
    mybins = np.arange(0, 256)
    n = len(loader.dataset)
    PRE = np.zeros((n, len(mybins) - 1))
    REC = np.zeros((n, len(mybins) - 1))
    F1 = np.zeros((n, len(mybins) - 1))
    MAE = np.zeros(n)
    IOU = np.zeros(n)
    val_loss_sum = 0.0
    val_seg_sum = 0.0
    n_iters = 0
    idx = 0
    sample_for_log = None
    sample_pred_for_log = None

    for batch_i, data in enumerate(pbar):
        if args.smoke_test and batch_i >= args.smoke_test:
            break
        names = data["image_name"]
        inputs, gt = data["image"], data["gt"]
        depth = data["depth"]
        if args.device != "cpu":
            inputs = inputs.to(args.device, non_blocking=True)
            gt = gt.to(args.device, non_blocking=True)
            depth = depth.to(args.device, non_blocking=True)

        pred_sig, _, loss, target_loss = model.eval_forward(inputs, depth, gt)
        val_loss_sum += float(loss.detach().cpu().item())
        val_seg_sum += float(target_loss.detach().cpu().item())
        n_iters += 1

        pred_sig = pred_sig.cpu()
        gt_cpu = gt.cpu()
        for k in range(inputs.shape[0]):
            p = pred_sig[k]
            if p.shape[-2:] != gt_cpu[k].shape[-2:]:
                p = F.upsample(p[None, ...], size=gt_cpu[k].shape[-2:], mode="bilinear")[0]
            pre, rec, f1 = f1score_torch(p, gt_cpu[k])
            mae = torch.nn.L1Loss()(p, gt_cpu[k].float()).item()
            PRE[idx + k, :] = pre.numpy().reshape(-1)
            REC[idx + k, :] = rec.numpy().reshape(-1)
            F1[idx + k, :] = f1.numpy().reshape(-1)
            MAE[idx + k] = mae
            IOU[idx + k] = iou_from_pred(p, gt_cpu[k])
        if sample_for_log is None:
            sample_for_log = {
                "image": inputs[:args.log_grid_n].cpu(),
                "gt": gt[:args.log_grid_n].cpu(),
                "depth": depth[:args.log_grid_n].cpu(),
                "image_name": names[:args.log_grid_n],
                "label": data.get("label", torch.empty(0))[:args.log_grid_n],
            }
            sample_pred_for_log = pred_sig[:args.log_grid_n]
        idx += inputs.shape[0]

    pbar.close()
    n_filled = max(idx, 1)
    PRE_m = PRE[:n_filled].mean(0)
    REC_m = REC[:n_filled].mean(0)
    f1_curve = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)

    metrics = {
        "loss": val_loss_sum / max(n_iters, 1),
        "seg_loss": val_seg_sum / max(n_iters, 1),
        "F1": float(np.amax(f1_curve)),
        "MAE": float(MAE[:n_filled].mean()),
        "IoU": float(IOU[:n_filled].mean()),
    }

    if writer is not None and sample_for_log is not None:
        log_augmented_batch(writer, sample_for_log, step=epoch,
                            tag="Val/inputs", n=args.log_grid_n)
        pred_grid = vutils.make_grid(
            to_3ch(sample_pred_for_log), nrow=args.log_grid_n, pad_value=1.0
        )
        writer.add_image("Val/predictions", pred_grid, global_step=epoch)
    return metrics


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def finetune_main(args):
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    setup_seed(args.seed)
    if args.smoke_test:
        print(f"[smoke_test] enabled: capping {args.smoke_test} batches per epoch, forcing --epochs 1")
        args.epochs = 1
        # don't snapshot the project tree on a smoke run
        args.COPY = False

    builders = {
        "csv": Mydataset.build_csv_dataset,
        "finetune": Mydataset.build_finetune_dataset,
        "dis": Mydataset.build_dataset,
    }
    builder = builders[args.dataset_builder]
    train_ds = builder(is_train=True, args=args)
    val_ds = builder(is_train=False, args=args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    print(f"[data] train={len(train_ds)} val={len(val_ds)}")
    print(f"[model] building {args.model} ...")
    model, model_name = build_model(args)
    load_pretrained(model, args.checkpoint)
    meta = setup_finetune_mode(model, args)
    print(f"[mode] {meta}")

    # Resume LoRA adapter weights from a previous run (after injection).
    resume_path = getattr(args, "resume_lora", "")
    if resume_path:
        if args.mode != "lora":
            raise ValueError("--resume_lora is only valid with --mode lora")
        ckpt = torch.load(resume_path, map_location="cpu")
        if not (isinstance(ckpt, dict) and ckpt.get("mode") == "lora" and "lora" in ckpt):
            raise ValueError(
                f"{resume_path} is not a LoRA checkpoint (expected "
                "{'mode': 'lora', 'lora': <state_dict>})"
            )
        lora_lib.load_lora_state_dict(model, ckpt["lora"], strict=True)
        print(f"[resume_lora] loaded {len(ckpt['lora'])} LoRA tensors from {resume_path}")

    model.to(device)

    optimizer = utiles.build_optimizer(args, model)
    lr_scheduler, EPOCH = create_scheduler(args, optimizer)
    scaler = GradScaler()

    train_time = str(datetime.datetime.today()).replace(" ", "_").replace(":", "_")[:-7]
    # Include the config file stem in the run tag so runs are grouped by their
    # config. Falls back to args.mode when --config is empty.
    config_stem = Path(getattr(args, "config", "") or "").stem
    run_id_part = config_stem or args.mode
    run_tag = f"{model_name}_{run_id_part}_{train_time}"
    ckpt_dir = os.path.join(args.checkpoints_save_path, run_tag)
    log_dir = os.path.join("runs", run_tag)
    valid_dir = os.path.join("valid_sample", run_tag)
    if not args.DEBUG:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        if args.COPY:
            os.makedirs(os.path.join(valid_dir, "project_copy"), exist_ok=True)
            copy_allfiles(os.getcwd(), os.path.join(valid_dir, "project_copy"))
    writer = None if args.DEBUG else SummaryWriter(log_dir=log_dir)

    best_metric = -1.0  # higher is better for F1; we'll invert sign below for MAE
    best_kind = args.eval_metric

    for epoch in range(EPOCH):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, args, epoch, writer)
        val_stats = validate(model, val_loader, args, epoch, writer)
        lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step(epoch)

        msg = (
            f"epoch {epoch+1}/{EPOCH}  "
            f"train_loss={train_stats['loss']:.4f}  train_seg={train_stats['seg_loss']:.4f}  "
            f"val_loss={val_stats['loss']:.4f}  val_seg={val_stats['seg_loss']:.4f}  "
            f"F1={val_stats['F1']:.4f}  MAE={val_stats['MAE']:.4f}  IoU={val_stats['IoU']:.4f}  "
            f"lr={lr:.2e}"
        )
        print(msg)

        if writer is not None:
            writer.add_scalar("Train/loss", train_stats["loss"], epoch + 1)
            writer.add_scalar("Train/seg_loss", train_stats["seg_loss"], epoch + 1)
            writer.add_scalar("Val/loss", val_stats["loss"], epoch + 1)
            writer.add_scalar("Val/seg_loss", val_stats["seg_loss"], epoch + 1)
            writer.add_scalar("Val/F1", val_stats["F1"], epoch + 1)
            writer.add_scalar("Val/MAE", val_stats["MAE"], epoch + 1)
            writer.add_scalar("Val/IoU", val_stats["IoU"], epoch + 1)
            writer.add_scalar("Optim/lr", lr, epoch + 1)

        # Track best & save
        score = val_stats[best_kind] if best_kind in val_stats else val_stats["F1"]
        is_better = (
            (best_kind == "F1" and score > best_metric)
            or (best_kind == "MAE" and (best_metric < 0 or score < best_metric))
            or best_metric < 0
        )
        if is_better and not args.DEBUG:
            best_metric = score
            best_path = os.path.join(
                ckpt_dir,
                f"best_{best_kind}_{score:.6f}_epoch{epoch+1}.pth",
            )
            save_checkpoint(model, best_path, args)
            Mydataset.keep_n_files(ckpt_dir, n=3)
        if not args.DEBUG:
            save_checkpoint(model, os.path.join(ckpt_dir, "LAST.pth"), args)

        torch.cuda.empty_cache()

    if writer is not None:
        writer.close()
    print(f"[done] best {best_kind} = {best_metric:.6f}; checkpoints in {ckpt_dir}")
