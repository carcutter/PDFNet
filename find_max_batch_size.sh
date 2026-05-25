#!/usr/bin/env bash
# Probe powers-of-2 batch sizes to find the largest one that fits on the GPU.
#
# How it works:
#   For each size in $PROBE_SIZES, spawn a *fresh* python subprocess running
#   Finetune_PDFNet.py with --smoke_test 2, which executes 2 train + 2 val
#   batches. That's enough to allocate all the activation, gradient and
#   optimizer-state tensors at the requested batch size — i.e. if the probe
#   succeeds, the real training will not OOM later at that size.
#
#   A fresh subprocess per attempt is required: PyTorch's CUDA caching
#   allocator does *not* release memory back to the OS on
#   `torch.cuda.empty_cache()` after an OOM. Only process exit does.
#
#   After every successful attempt, the file MAX_BATCH_SIZE_BY_GPU.md is
#   rewritten with the new known-good (GPU, mode, batch_size, peak_VRAM)
#   tuple. If the loop crashes or is killed, the latest known-good survives.
#
# Usage:
#   ./find_max_batch_size.sh                                 # default: --mode full
#   PROBE_MODE=lora ./find_max_batch_size.sh                 # probe LoRA
#   PROBE_SIZES="2 4 8 16" ./find_max_batch_size.sh          # custom sweep
#   ./find_max_batch_size.sh --lora_rank 32 --lora_alpha 16  # extra args -> python
#
# Caveats:
#   - If another GPU process is running, the probe sees its memory as baseline
#     and the result will be a lower-bound on the true max. The script warns
#     when it detects >2 GiB in use at start.
#   - Probing is GPU-only. CPU OOMs are not handled specially.

set -uo pipefail

if [[ -z "${VIRTUAL_ENV:-}" && -d .venv ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

PROBE_MODE=${PROBE_MODE:-full}
PROBE_SIZES=${PROBE_SIZES:-"1 2 4 8 16 32 64 128"}
CKPT=${CKPT:-checkpoints/PDFNet_Best.pth}
MD=MAX_BATCH_SIZE_BY_GPU.md
WORK=logs/batch_size_probe
mkdir -p "$WORK"
PASSTHROUGH=("$@")

# ---- prereq checks ---------------------------------------------------------
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: pretrained checkpoint missing at $CKPT"
    echo "       Run: python download_checkpoint.py"
    exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not in PATH"; exit 2
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_USED_NOW=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

if (( GPU_USED_NOW > 2048 )); then
    echo "WARN: GPU already using ${GPU_USED_NOW} MiB / ${GPU_TOTAL} MiB."
    echo "      The probe will see this as baseline; results will be a lower bound."
    echo "      Stop other GPU processes for an accurate measurement."
    echo
fi
echo "GPU:        $GPU_NAME ($GPU_TOTAL MiB)"
echo "Mode:       $PROBE_MODE"
echo "Sizes:      $PROBE_SIZES"
echo "Pass-thru:  ${PASSTHROUGH[*]:-(none)}"
echo

# ---- MD file boilerplate ---------------------------------------------------
if [[ ! -f "$MD" ]]; then
    cat > "$MD" <<'EOF'
# Max batch size by GPU

Probed via `find_max_batch_size.sh` — fresh python subprocess per attempt,
running 2 train + 2 validation steps so every activation, gradient and
optimizer-state tensor is actually allocated.

The row for each `(GPU, mode)` pair is rewritten after every successful
attempt, so the latest known-good survives a crash mid-probe.

| GPU | Total VRAM (MiB) | Mode | Max batch | Peak VRAM at max (MiB) | Probed at |
|---|---|---|---|---|---|
EOF
fi

write_max() {
    local size=$1 peak=$2
    # Match the (GPU, mode) prefix so the same row is replaced on re-runs.
    local key="| $GPU_NAME | $GPU_TOTAL | $PROBE_MODE |"
    {
        grep -vF "$key" "$MD" || true
        echo "$key $size | $peak | $(date -u +%Y-%m-%dT%H:%M:%SZ) |"
    } > "$MD.tmp"
    mv "$MD.tmp" "$MD"
}

# ---- probe a single batch size --------------------------------------------
probe() {
    local size=$1
    local tag="bs_${PROBE_MODE}_${size}"
    local log="$WORK/${tag}.log"
    local mem_log="$WORK/${tag}.mem"
    local ckpt_dir="$WORK/ckpt_${PROBE_MODE}_${size}"
    rm -rf "$ckpt_dir" "$mem_log"
    : > "$mem_log"

    echo "--- probing batch_size=$size (mode=$PROBE_MODE) ---"

    # Background memory sampler (every 0.5s). Captures peak even if process dies.
    (
        while true; do
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                | head -1 >> "$mem_log"
            sleep 0.5
        done
    ) &
    local sampler=$!

    local rc=0
    python -u Finetune_PDFNet.py \
        --mode "$PROBE_MODE" \
        --checkpoint "$CKPT" \
        --synthesize_depth \
        --smoke_test 2 \
        --batch_size "$size" \
        --num_workers 0 \
        --checkpoints_save_path "$ckpt_dir" \
        --COPY False \
        "${PASSTHROUGH[@]}" \
        > "$log" 2>&1
    rc=$?

    kill "$sampler" 2>/dev/null
    wait "$sampler" 2>/dev/null

    local peak
    peak=$(sort -n "$mem_log" 2>/dev/null | tail -1)
    peak=${peak:-0}

    # Clean up probe checkpoints; tensorboard dirs in runs/ are left for the
    # user to inspect / delete.
    rm -rf "$ckpt_dir"

    if [[ $rc -eq 0 ]]; then
        echo "  OK    batch=$size  peak_mem=${peak} MiB  (headroom: $((GPU_TOTAL - peak)) MiB)"
        write_max "$size" "$peak"
        return 0
    else
        local reason="rc=$rc"
        if grep -qE "out of memory|CUDA out of memory|OutOfMemoryError" "$log"; then
            reason="OOM"
        elif [[ $rc -eq 137 || $rc -eq 139 ]]; then
            reason="killed (likely OOM)"
        fi
        echo "  FAIL  batch=$size  ${reason}  peak_mem=${peak} MiB"
        echo "        last lines of $log:"
        tail -5 "$log" | sed 's/^/        | /'
        return 1
    fi
}

# ---- main loop -------------------------------------------------------------
echo
last_ok=""
for size in $PROBE_SIZES; do
    if probe "$size"; then
        last_ok=$size
    else
        echo
        echo "Stopping at first failure (batch_size=$size)."
        break
    fi
done

echo
echo "===================="
if [[ -n "$last_ok" ]]; then
    echo "Max batch size on $GPU_NAME (mode=$PROBE_MODE): $last_ok"
else
    echo "No batch size succeeded — first probe failed. Inspect $WORK/."
fi
echo "MD file:  $MD"
echo "===================="
cat "$MD"
