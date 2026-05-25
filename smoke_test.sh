#!/usr/bin/env bash
# Smoke test: run a few train+val batches in both LoRA and full mode to
# verify the full fine-tuning pipeline (data, model, checkpoint load, LoRA
# injection, forward+backward, AMP, val metrics, tensorboard, checkpoint save).
#
# Takes ~1-2 minutes per mode on an L40S. Exits 0 on success.

set -uo pipefail

# Activate uv venv if it's there and not already active
if [[ -z "${VIRTUAL_ENV:-}" && -d .venv ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

CKPT=checkpoints/PDFNet_Best.pth
SMOKE_DIR=checkpoints/smoke_test
SMOKE_RUNS=runs/smoke_test
LOG_DIR=logs/smoke_test
BATCHES=3

mkdir -p "$LOG_DIR"
# clear previous smoke artifacts so we test fresh saves
rm -rf "$SMOKE_DIR" "$SMOKE_RUNS"

if [[ ! -f "$CKPT" ]]; then
    echo "FAIL: pretrained checkpoint missing at $CKPT"
    echo "      Run: python download_checkpoint.py"
    exit 2
fi
if [[ ! -f data/index.csv ]]; then
    echo "FAIL: dataset CSV missing at data/index.csv"
    echo "      Run: python build_dataset_csv.py"
    exit 2
fi

run_mode() {
    local mode=$1; shift
    local log="$LOG_DIR/${mode}.log"
    local ts; ts=$(date +%H:%M:%S)
    echo
    echo "============================================================"
    echo "[$ts] smoke: --mode $mode   batches=$BATCHES   log=$log"
    echo "============================================================"

    # We tee the live output AND keep an exit code from the python run
    set -o pipefail
    python -u Finetune_PDFNet.py \
        --mode "$mode" \
        --checkpoint "$CKPT" \
        --synthesize_depth \
        --smoke_test "$BATCHES" \
        --num_workers 0 \
        --batch_size 1 \
        --checkpoints_save_path "$SMOKE_DIR" \
        "$@" \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}

    echo
    echo "--- assertions for '$mode' ---"
    local pass=1
    assert_line() {
        local pattern=$1 description=$2
        if grep -qE "$pattern" "$log"; then
            echo "  ✓ $description"
        else
            echo "  ✗ $description (pattern: $pattern)"
            pass=0
        fi
    }
    [[ $rc -eq 0 ]] && echo "  ✓ exit code 0" || { echo "  ✗ exit code $rc"; pass=0; }
    assert_line '\[checkpoint\] loaded ' 'pretrained checkpoint loaded'
    assert_line '\[mode\] ' 'mode setup printed'
    assert_line '\[csv-dataset\] ' 'CSV dataset built'
    assert_line 'train 1/1' 'training loop ran'
    assert_line 'val   1/1|val_loss=' 'validation loop ran'
    assert_line 'F1=' 'F1 metric computed'
    assert_line '\[done\] best' 'training finished cleanly'

    # smoke-test mode triggers checkpoint save IF val improved → it always does on epoch 1
    if find "$SMOKE_DIR" -name '*.pth' 2>/dev/null | grep -q .; then
        echo "  ✓ checkpoint file written under $SMOKE_DIR"
    else
        echo "  ✗ no checkpoint file produced"
        pass=0
    fi
    # SummaryWriter writes to runs/<model>_<mode>_<ts>/; we move it under
    # $SMOKE_RUNS after this assertion runs, so look at the live location.
    if find runs/ -maxdepth 2 -path "*_${mode}_*" -name 'events.out.tfevents.*' 2>/dev/null | grep -q .; then
        echo "  ✓ tensorboard event file written"
    else
        echo "  ✗ no tensorboard event file in runs/*_${mode}_*"
        pass=0
    fi

    [[ $pass -eq 1 ]] && echo "PASS: $mode" || { echo "FAIL: $mode"; return 1; }
}

# pyproject pin / sanity: torch with CUDA available
python - <<'PY' || { echo 'FAIL: torch/CUDA not ready'; exit 2; }
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  gpu {torch.cuda.get_device_name(0)}")
PY

# Override default --output_dir so we can clean it up easily
export PDFNET_SMOKE_RUNS="$SMOKE_RUNS"

# We can't easily redirect the SummaryWriter dir from the CLI without code changes,
# so the writer goes to runs/ as usual. Sniff which subdir was just created and
# count it as the smoke artifact for the assertion above.
RUNS_BEFORE=$(ls runs/ 2>/dev/null | wc -l)

overall=0
run_mode lora --lora_rank 8 --lora_alpha 16 || overall=1

# Locate the run dir that just appeared, treat it as the smoke-test artifact
new_dirs=$(ls -dt runs/PDFNet_swinB_lora_* 2>/dev/null | head -1)
[[ -n "$new_dirs" ]] && { mkdir -p "$SMOKE_RUNS"; mv "$new_dirs" "$SMOKE_RUNS/lora/"; }

run_mode full || overall=1
new_dirs=$(ls -dt runs/PDFNet_swinB_full_* 2>/dev/null | head -1)
[[ -n "$new_dirs" ]] && { mkdir -p "$SMOKE_RUNS"; mv "$new_dirs" "$SMOKE_RUNS/full/"; }

echo
echo "============================================================"
if [[ $overall -eq 0 ]]; then
    echo "ALL SMOKE TESTS PASSED"
    echo "  artifacts:  $SMOKE_DIR  $SMOKE_RUNS  $LOG_DIR"
    exit 0
else
    echo "SMOKE TESTS FAILED — inspect logs under $LOG_DIR"
    exit 1
fi
