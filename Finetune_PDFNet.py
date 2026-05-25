"""Fine-tune PDFNet on a user-provided dataset.

Expected layout (default):

    ./data/
      train/
        images/    *.jpg|png|...
        masks/     *.png  (binary GT)
        depth/     *.png  (pseudo-depth from DAM-V2; single dir or split into
                          depth_small/depth_base/depth_large/depth_large_1024)
      val/
        images/
        masks/
        depth/

Quickstart:

    # 1. one-time: grab pretrained PDFNet + Swin-B backbone
    python download_checkpoint.py

    # 2. full fine-tune
    python Finetune_PDFNet.py --checkpoint checkpoints/PDFNet_Best.pth

    # 3. LoRA fine-tune (much smaller checkpoints, faster, less VRAM)
    python Finetune_PDFNet.py --mode lora --checkpoint checkpoints/PDFNet_Best.pth \\
                              --lora_rank 8 --lora_alpha 16

TensorBoard logs (under ./runs/<run_tag>/):
    Train/loss, Train/seg_loss
    Val/loss, Val/seg_loss, Val/F1, Val/MAE, Val/IoU
    Inputs/augmented   - input batch after augmentations (image|GT|depth|overlay)
    Val/inputs         - one batch of val inputs each epoch
    Val/predictions    - model predictions for that batch
"""
import argparse
from pathlib import Path

from finetune_main import finetune_main


def get_args_parser():
    p = argparse.ArgumentParser("PDFNet fine-tuning", add_help=False)

    # ---- core training ----
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--epochs", default=30, type=int)
    p.add_argument("--update_freq", default=1, type=int,
                   help="gradient accumulation steps; 1 disables")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--device", default="cuda")

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

    # ---- fine-tune specific ----
    p.add_argument("--checkpoint", default="checkpoints/PDFNet_Best.pth", type=str,
                   help="path to the pretrained PDFNet .pth (loaded with strict=False)")
    p.add_argument("--mode", default="full", choices=["full", "lora"],
                   help="full = train all params (optionally with --freeze_backbone); "
                        "lora = freeze base, only train injected LoRA adapters")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="(full mode) freeze the Swin-B encoder, fine-tune decoders only")
    # LoRA
    p.add_argument("--lora_rank", default=8, type=int)
    p.add_argument("--lora_alpha", default=16.0, type=float)
    p.add_argument("--lora_dropout", default=0.0, type=float)
    p.add_argument("--lora_targets", nargs="*", default=None,
                   help="fnmatch patterns for module names to wrap. Default: decoder.* "
                        "and depth_decoder.*. Example: --lora_targets 'decoder.FSE_mix.*'")
    p.add_argument("--resume_lora", default="", type=str,
                   help="(lora mode) path to a previous LoRA checkpoint "
                        "({'mode':'lora','lora':<state_dict>}). Loaded *after* LoRA injection. "
                        "Note: only the adapter weights resume; optimizer + scheduler restart.")

    # ---- data ----
    p.add_argument("--data_path", default="./data", type=str)
    p.add_argument("--dataset_builder", default="csv",
                   choices=["csv", "finetune", "dis"],
                   help="'csv'      = read pairs from --csv_path with 80/20 split (default); "
                        "'finetune' = ./data/{train,val}/{images,masks,depth} layout; "
                        "'dis'      = original DIS-5K layout under args.data_path")
    # csv-mode
    p.add_argument("--csv_path", default="data/index.csv", type=str,
                   help="CSV with image_path,mask_path,depth_path,dataset columns. "
                        "Generate with `python build_dataset_csv.py`.")
    p.add_argument("--val_split", default=0.2, type=float,
                   help="(csv mode) validation fraction")
    p.add_argument("--csv_split_seed", default=42, type=int,
                   help="(csv mode) RNG seed for the 80/20 shuffle; same seed gives "
                        "the same split across runs.")
    p.add_argument("--synthesize_depth", action="store_true",
                   help="if no depth file is found, synthesize pseudo-depth from "
                        "RGB grayscale. Enable until DAM-V2 depth maps are generated.")
    # finetune-mode
    p.add_argument("--train_subdir", default="train", type=str)
    p.add_argument("--val_subdir", default="val", type=str)
    p.add_argument("--depth_variants", nargs="*",
                   default=["depth_large", "depth_base", "depth_small"],
                   help="depth-input variant dirs to sample from during training. "
                        "Pass --depth_variants depth for a single-variant setup.")
    p.add_argument("--depth_gt_dir", default="depth_large_1024", type=str)
    p.add_argument("--depth_fallback_dir", default="depth", type=str,
                   help="used when a per-variant depth dir doesn't exist on disk")
    p.add_argument("--labels_from_filename", action="store_true",
                   help="parse one-hot labels from DIS-5K '#'-separated filenames; "
                        "off by default for custom data")
    p.add_argument("--chached", default=False, type=bool)

    # ---- optimization ----
    p.add_argument("--opt", default="adamw", type=str)
    p.add_argument("--opt_eps", default=1e-8, type=float)
    p.add_argument("--opt_betas", default=None, type=float, nargs="+")
    p.add_argument("--clip_grad", default=None, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--sched", default="cosine", type=str)
    p.add_argument("--lr", default=1e-5, type=float)
    p.add_argument("--warmup_lr", default=1e-6, type=float)
    p.add_argument("--min_lr", default=1e-6, type=float)
    p.add_argument("--warmup_epochs", default=2, type=int)
    p.add_argument("--decay_epochs", default=300, type=float)
    p.add_argument("--cooldown_epochs", default=0, type=int)
    p.add_argument("--patience_epochs", default=10, type=int)
    p.add_argument("--decay_rate", "--dr", default=0.1, type=float)

    # ---- logging ----
    p.add_argument("--checkpoints_save_path", default="checkpoints/finetune", type=str)
    p.add_argument("--output_dir", default="")
    p.add_argument("--COPY", default=True, type=bool,
                   help="snapshot the current working tree into valid_sample/<run>/project_copy/")
    p.add_argument("--log_batch_every", default=1, type=int,
                   help="log an augmented-batch grid every N epochs (1=every epoch)")
    p.add_argument("--log_grid_n", default=4, type=int,
                   help="number of samples in the augmented-batch grid")
    p.add_argument("--eval_metric", default="F1", choices=["F1", "MAE"],
                   help="metric used to decide 'best' checkpoint")
    p.add_argument("--DEBUG", default=False, type=bool,
                   help="disables checkpoint saving and tensorboard writes")
    p.add_argument("--smoke_test", default=0, type=int,
                   help="if >0, cap train AND val to this many batches per epoch and "
                        "force --epochs 1. Used by smoke_test.sh to validate the full "
                        "training pipeline end-to-end in ~1 minute.")
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PDFNet fine-tune script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=False, exist_ok=True)
    finetune_main(args)
