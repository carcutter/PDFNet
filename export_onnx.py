"""Export a (fine-tuned) PDFNet checkpoint to ONNX.

Wraps `PDFNet_process.inference(image, depth) -> sigmoid_mask` and traces it.
The base PDFNet checkpoint is loaded first, then the fine-tuned weights on top
(LoRA adapters or a full state dict, selected by --mode) — exactly the same
load path as run_inference.py, so the ONNX graph matches what that script runs.

Fixed input shape: the patch split/merge logic is tied to the spatial size, so
inputs are exported at a static (1, 3/1, input_size, input_size). --dynamic_batch
marks axis 0 dynamic (best effort; the patch arithmetic is traced at batch=1).

Inputs:
  image: (1, 3, H, W) float32, ImageNet-normalized (same as the dataloader).
  depth: (1, 1, H, W) float32 (the model min/max-normalizes + expands to 3ch).
Output:
  mask:  (1, 1, h, w) float32 sigmoid probability map (h,w = model output res).

Quickstart:

  python export_onnx.py --config config/training/full_decoder.yaml \\
      --finetune_checkpoint checkpoints/finetune/PDFNet_swinB_full_decoder_.../best_*.pth

  python export_onnx.py \\
      --finetune_checkpoint checkpoints/finetune/PDFNet_swinB_lora_.../LAST.pth \\
      --output exports/pdfnet_lora.onnx
"""
import argparse
import os
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.PDFNet import build_model
from models import lora as lora_lib
from finetune_main import load_pretrained
from run_inference import get_args_parser as base_args_parser
from run_inference import _apply_yaml_defaults, _load_full_ckpt, _load_lora_ckpt


class InferenceWrapper(nn.Module):
    """ONNX-friendly forward: (image, depth) -> sigmoid mask."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image, depth):
        sigmoid, _ = self.model.inference(image, depth)
        return sigmoid


class _SDPAAttn(nn.Module):
    """Drop-in for nn.MultiheadAttention(num_heads=1, bias=False, batch_first=True).

    CoA calls `self.Att(q, kv, kv)[0]`. torch.export can't functionalize MHA's
    packed-projection (`_in_projection_packed` -> chunk/as_strided/copy_to), so
    we reimplement the same maths with scaled_dot_product_attention, sharing the
    original weights. Returns (out, None) to keep the `[0]` indexing working.
    """

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        assert mha.num_heads == 1, f"expected 1 head, got {mha.num_heads}"
        assert mha.in_proj_bias is None, "expected bias=False"
        self.embed_dim = mha.embed_dim
        self.in_proj_weight = mha.in_proj_weight  # (3E, E), shared Parameter
        self.out_proj = mha.out_proj              # nn.Linear(E, E, bias=False)

    def forward(self, q, k, v, *args, **kwargs):
        E = self.embed_dim
        w = self.in_proj_weight
        Q = q @ w[:E].t()
        K = k @ w[E:2 * E].t()
        V = v @ w[2 * E:].t()
        out = F.scaled_dot_product_attention(Q, K, V)  # single head, scale=1/sqrt(E)
        return self.out_proj(out), None


def patch_mha_for_export(model: nn.Module) -> int:
    """Replace every nn.MultiheadAttention with the SDPA equivalent. In eval mode
    (dropout off) this is numerically equivalent; the parity check confirms it."""
    n = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                setattr(module, name, _SDPAAttn(child))
                n += 1
    return n


def load_sample_input(args, device):
    """Build a real (image, depth) input from --sample_image, preprocessed exactly
    like the dataloader (bilinear resize -> /255 -> ImageNet-normalize; depth is
    grayscale, which the model re-normalizes internally). Falls back to random
    noise if no sample image is available."""
    s = args.input_size
    path = args.sample_image
    if not path or not os.path.isfile(path):
        print(f"[verify] no sample image at {path!r}; using random input")
        return (torch.randn(1, 3, s, s, device=device),
                torch.rand(1, 1, s, s, device=device), False)

    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = torch.from_numpy(rgb).permute(2, 0, 1)[None, ...].float()
    im = F.interpolate(im, size=[s, s], mode="bilinear", align_corners=True)[0] / 255.0
    gray = (0.299 * im[0] + 0.587 * im[1] + 0.114 * im[2])[None, None, ...]  # (1,1,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = ((im - mean) / std)[None, ...].to(device)
    depth = gray.to(device)
    print(f"[verify] sample image: {path}")
    return image, depth, True


def build_and_load(args) -> nn.Module:
    print(f"[model] building {args.model} ...")
    # Swap the encoder for the ONNX-friendly Swin (functional SW-MSA mask, no
    # in-place slice assignment) before build_model resolves the SwinB symbol.
    # Weights are identical, so the fine-tuned checkpoint loads unchanged.
    import models.PDFNet as pdfnet_mod
    from models.swin_transformer_onnx import SwinB as SwinB_onnx
    pdfnet_mod.SwinB = SwinB_onnx
    print("[model] using swin_transformer_onnx.SwinB encoder for export")

    # The model uses the deprecated F.upsample(..., align_corners=True) via
    # _upsample_/_upsample_like. Its bilinear-upsample decomposition injects a
    # non-functional copy_to/as_strided that run_decompositions() rejects.
    # F.interpolate is numerically identical (same args) and decomposes cleanly.
    import models.utils as utils_mod

    def _upsample_like_onnx(src, tar, mode="bilinear"):
        if mode == "bilinear":
            return F.interpolate(src, size=tar.shape[2:], mode=mode, align_corners=True)
        return F.interpolate(src, size=tar.shape[2:], mode=mode)

    def _upsample_onnx(src, size, mode="bilinear"):
        if mode == "bilinear":
            return F.interpolate(src, size=size, mode=mode, align_corners=True)
        return F.interpolate(src, size=size, mode=mode)

    for mod in (utils_mod, pdfnet_mod):
        mod._upsample_ = _upsample_onnx
        mod._upsample_like = _upsample_like_onnx
    print("[model] patched _upsample_/_upsample_like to F.interpolate for export")

    # FSE.get_boundary builds avg_pool2d kernel/padding from pred.shape//8. Under
    # the legacy tracer these become dynamic aten::size ops, and avg_pool2d needs
    # constant kernel sizes -> export fails (PDFNet.py:146). int() bakes them as
    # constants (shapes are static for a fixed input). Identical numerically.
    def _get_boundary_onnx(self, pred):
        h = int(pred.shape[-2]); w = int(pred.shape[-1])
        kh, kw = h // 8, w // 8
        ph, pw = kh // 2, kw // 2
        ks = (kh + 1, kw + 1) if (kh % 2 == 0) else (kh, kw)
        s = pred.sigmoid()
        return abs(s - F.avg_pool2d(s, kernel_size=ks, stride=1, padding=(ph, pw)))

    pdfnet_mod.FSE.get_boundary = _get_boundary_onnx
    print("[model] patched FSE.get_boundary to constant kernel sizes for export")

    # FSE.forward calls F.adaptive_avg_pool2d(x, output_size=[H//pi, W//pi]) with
    # output_size derived from dynamic shapes; the legacy exporter needs it
    # constant. Coerce to Python ints (baked under the static-shape trace).
    # ONNX has no adaptive pooling; the legacy exporter only converts it when the
    # input size is statically known, which it isn't under the trace. Rewrite it
    # as a plain avg_pool2d with kernel/stride computed from the (constant, baked
    # via int()) input/output sizes. Exact for divisible sizes (FSE pool_ratios
    # divide evenly); falls back to the original otherwise.
    _orig_aap = F.adaptive_avg_pool2d

    def _aap_const(x, output_size):
        Hin, Win = int(x.shape[-2]), int(x.shape[-1])
        if isinstance(output_size, (list, tuple)):
            Hout, Wout = int(output_size[0]), int(output_size[1])
        else:
            Hout = Wout = int(output_size)
        if Hout > 0 and Wout > 0 and Hin % Hout == 0 and Win % Wout == 0:
            kh, kw = Hin // Hout, Win // Wout
            return F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
        return _orig_aap(x, [Hout, Wout])

    F.adaptive_avg_pool2d = _aap_const
    pdfnet_mod.F.adaptive_avg_pool2d = _aap_const
    print("[model] patched F.adaptive_avg_pool2d -> avg_pool2d (constant kernel) for export")

    model, _ = build_model(args)
    load_pretrained(model, args.checkpoint)

    if args.mode == "lora":
        targets = list(args.lora_targets) if args.lora_targets else list(lora_lib.DEFAULT_DECODER_TARGETS)
        wrapped = lora_lib.inject_lora(model, targets=targets, rank=args.lora_rank,
                                       alpha=args.lora_alpha, dropout=args.lora_dropout)
        if not wrapped:
            raise RuntimeError(f"No modules matched LoRA targets {targets}.")
        n = _load_lora_ckpt(model, args.finetune_checkpoint)
        print(f"[lora] wrapped {len(wrapped)} modules, loaded {n} LoRA tensors "
              f"from {args.finetune_checkpoint}")
    elif args.mode == "full":
        _load_full_ckpt(model, args.finetune_checkpoint)
    else:
        raise ValueError(f"Unknown --mode {args.mode}")
    return model


def main():
    parser = argparse.ArgumentParser("PDFNet ONNX export", parents=[base_args_parser()])
    parser.add_argument("--output", default="", type=str,
                        help="output .onnx path. Defaults to "
                             "exports/<finetune-checkpoint-dir-name>.onnx")
    parser.add_argument("--opset", default=17, type=int,
                        help="ONNX opset version (>=14 needed for MultiheadAttention).")
    parser.add_argument("--dynamic_batch", action="store_true",
                        help="mark batch axis dynamic (patch logic is traced at "
                             "batch=1; verify before relying on batched inference).")
    parser.add_argument("--legacy", action="store_true",
                        help="use the legacy TorchScript ONNX exporter instead of "
                             "the dynamo-based one. The TorchScript path chokes on "
                             "this model's index_put inside a subblock, so dynamo "
                             "is the default.")
    parser.add_argument("--sample_image", default="5#Artifact#1#Basket#3342299538_59c014a904_o.jpg",
                        type=str,
                        help="real image used for the parity check (preprocessed "
                             "like the dataloader). Falls back to random noise if "
                             "the file doesn't exist.")
    parser.add_argument("--verify_threshold", default=1e-2, type=float,
                        help="if the max abs difference between the PyTorch and "
                             "ONNX mask exceeds this, print a WARNING.")
    parser.add_argument("--no_verify", action="store_true",
                        help="skip the onnxruntime parity check against PyTorch.")
    # Export on CPU by default: avoids competing with any running GPU job and
    # sidesteps fp16/cuda-only quirks during tracing. Override with --device cuda.
    parser.set_defaults(device="cpu")
    # Two-pass parse so --config defaults fold in before the final parse.
    prelim, _ = parser.parse_known_args()
    _apply_yaml_defaults(parser, prelim.config)
    # Re-assert CPU default: the YAML may carry device=cuda from training.
    parser.set_defaults(device="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    exp_name = Path(args.finetune_checkpoint).parent.name
    out_path = args.output or os.path.join("exports", f"{exp_name}.onnx")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    model = build_and_load(args).to(device).eval()
    # In-place activations (make_crs uses nn.SiLU(inplace=True)) leave dead
    # copy_to/as_strided nodes that dynamo's functionalization rejects. Disabling
    # inplace is mathematically identical — it only costs a little memory.
    n_inplace = 0
    for m in model.modules():
        if getattr(m, "inplace", False):
            m.inplace = False
            n_inplace += 1
    if n_inplace:
        print(f"[export] disabled inplace on {n_inplace} activation modules for tracing")

    # Build the real sample input up front: it's used both as the export example
    # and for the parity check.
    image, depth, is_real = load_sample_input(args, device)

    # Reference output from the ORIGINAL (unexported, unpatched) model — this is
    # what we compare the ONNX model against.
    wrapper = InferenceWrapper(model).to(device).eval()
    with torch.no_grad():
        torch_ref = wrapper(image, depth).cpu()

    # Swap MHA -> SDPA so torch.export can functionalize the graph.
    n_mha = patch_mha_for_export(model)
    if n_mha:
        print(f"[export] replaced {n_mha} MultiheadAttention modules with SDPA for export")
    wrapper = InferenceWrapper(model).to(device).eval()

    exporter = "legacy TorchScript" if args.legacy else "dynamo"
    print(f"[export] tracing inference at input {tuple(image.shape)} (+depth {tuple(depth.shape)}) "
          f"on {device}, opset {args.opset}, {exporter} exporter ...")

    with torch.no_grad():
        if args.legacy:
            dynamic_axes = None
            if args.dynamic_batch:
                dynamic_axes = {"image": {0: "batch"}, "depth": {0: "batch"}, "mask": {0: "batch"}}
            torch.onnx.export(
                wrapper, (image, depth), out_path,
                input_names=["image", "depth"], output_names=["mask"],
                opset_version=args.opset, do_constant_folding=True,
                dynamic_axes=dynamic_axes,
            )
        else:
            dynamic_shapes = None
            if args.dynamic_batch:
                batch = torch.export.Dim("batch")
                dynamic_shapes = {"image": {0: batch}, "depth": {0: batch}}
            # torch.export -> run_decompositions() ourselves (the default decomp
            # table is functional here), then hand the already-decomposed program
            # to the ONNX exporter. The exporter's own decomposition pass injects
            # a dead copy_to/as_strided (torch 2.8 bug); pre-decomposing avoids it.
            ep = torch.export.export(wrapper, (image, depth),
                                     dynamic_shapes=dynamic_shapes, strict=False)
            ep = ep.run_decompositions()
            torch.onnx.export(
                ep, (image, depth), out_path,
                input_names=["image", "depth"], output_names=["mask"],
                opset_version=args.opset, dynamo=True,
            )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[export] wrote {out_path} ({size_mb:.1f} MB)")

    # Structural check
    try:
        import onnx
        onnx.checker.check_model(onnx.load(out_path))
        print("[verify] onnx.checker passed")
    except Exception as e:
        print(f"[verify] onnx.checker WARNING: {e}")

    if args.no_verify:
        return

    # Numerical parity: original PyTorch model vs exported ONNX, on the same input.
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not available; skipping parity check")
        return

    torch_out = torch_ref.numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["mask"], {
        "image": image.cpu().numpy(),
        "depth": depth.cpu().numpy(),
    })[0]

    if torch_out.shape != ort_out.shape:
        print(f"[verify] WARNING: shape mismatch torch={torch_out.shape} onnx={ort_out.shape}")
        return

    diff = np.abs(torch_out - ort_out)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    # Segmentation-relevant metric: agreement of the binarized masks at 0.5.
    pt_bin = torch_out >= 0.5
    on_bin = ort_out >= 0.5
    inter = np.logical_and(pt_bin, on_bin).sum()
    union = np.logical_or(pt_bin, on_bin).sum()
    mask_iou = float(inter / union) if union > 0 else 1.0

    src = "sample image" if is_real else "random input"
    print(f"[verify] PyTorch vs ONNX on {src}: "
          f"max_abs_diff={max_abs:.3e} mean_abs_diff={mean_abs:.3e} mask_IoU@0.5={mask_iou:.4f}")
    if max_abs > args.verify_threshold:
        print(f"[verify] WARNING: max_abs_diff {max_abs:.3e} exceeds threshold "
              f"{args.verify_threshold:.3e} — the ONNX model diverges from PyTorch. "
              f"Inspect before deploying (try --opset 18, or --legacy).")
    else:
        print(f"[verify] OK: within threshold {args.verify_threshold:.3e}")


if __name__ == "__main__":
    main()
