"""Minimal LoRA adapters for PDFNet fine-tuning.

Supports wrapping nn.Linear and nn.Conv2d. The base layer is frozen; only
the low-rank A/B factors are trainable. Effective update is

    out = base(x) + (alpha / r) * lora_B(lora_A(x))

with lora_B initialised to zero so the wrapped model is bit-identical to
the original at step 0.
"""
import fnmatch
import math
import re
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


class LoRAConv2d(nn.Module):
    """LoRA for Conv2d. lora_A keeps the base conv's spatial kernel; lora_B is 1x1."""

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        if base.groups != 1:
            raise ValueError(
                "LoRAConv2d only supports groups=1 convs; "
                f"got groups={base.groups} for {base}"
            )
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Conv2d(
            base.in_channels, rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.lora_B = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


def _name_matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module):
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _parent_module(root: nn.Module, qualified_name: str) -> nn.Module:
    parent = root
    for part in qualified_name.split(".")[:-1]:
        parent = getattr(parent, part)
    return parent


def inject_lora(
    model: nn.Module,
    targets: Iterable[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    include_linear: bool = True,
    include_conv2d: bool = True,
) -> List[str]:
    """Replace matching Linear/Conv2d modules with LoRA-wrapped versions.

    `targets` is a list of fnmatch-style glob patterns matched against the
    fully-qualified module name (e.g. "decoder.FSE_mix.*", "decoder.*.w1").
    Returns the list of names that were wrapped.

    Skips modules whose direct parent is `nn.MultiheadAttention`: that class's
    forward accesses `self.out_proj.weight` directly, which would break if the
    child Linear were swapped for a LoRA wrapper.
    """
    targets = list(targets)
    candidates: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            continue  # already wrapped
        if include_linear and isinstance(module, nn.Linear):
            if _name_matches(name, targets):
                candidates.append((name, module))
        elif include_conv2d and isinstance(module, nn.Conv2d):
            if module.groups != 1:
                continue
            if _name_matches(name, targets):
                candidates.append((name, module))

    wrapped = []
    for name, module in candidates:
        parent = _parent_module(model, name)
        if isinstance(parent, nn.MultiheadAttention):
            continue  # see docstring
        if isinstance(module, nn.Linear):
            new_mod = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        else:
            new_mod = LoRAConv2d(module, rank=rank, alpha=alpha, dropout=dropout)
        _set_submodule(model, name, new_mod)
        wrapped.append(name)
    return wrapped


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze every parameter except LoRA A/B factors.

    bias = "none"    : biases also frozen (default)
    bias = "lora_only": biases of wrapped layers stay trainable
    bias = "all"     : all biases trainable
    """
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
        elif bias == "all" and name.endswith(".bias"):
            p.requires_grad = True
        else:
            p.requires_grad = False
    if bias == "lora_only":
        for module in model.modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                if getattr(module.base, "bias", None) is not None:
                    module.base.bias.requires_grad = True


def lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters for compact checkpoints."""
    out = {}
    for name, p in model.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            out[name] = p
    return out


def load_lora_state_dict(model: nn.Module, state_dict: dict, strict: bool = True) -> None:
    """Load LoRA-only state into a model that already has LoRA injected."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    expected = {n for n in model.state_dict() if "lora_A" in n or "lora_B" in n}
    loaded = set(state_dict.keys()) & expected
    missing_lora = expected - loaded
    if strict and missing_lora:
        raise RuntimeError(
            f"Missing {len(missing_lora)} LoRA parameters when loading; "
            f"example: {next(iter(missing_lora))}"
        )


def count_trainable(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# Sensible defaults: LoRA-ify everything in the decoder / depth decoder
# while keeping the encoder frozen. Targets the SwiGLU linears, MHA out_proj,
# and the make_crs / Bside / channel_mix convs inside the decoders.
DEFAULT_DECODER_TARGETS: Tuple[str, ...] = (
    "decoder.*",
    "depth_decoder.*",
)
