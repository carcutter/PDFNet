#!/usr/bin/env bash
# One-shot environment preparation for PDFNet fine-tuning on interior-segmentation data.
#
# Walks through the README steps that *can* be automated:
#   1. Create a uv-managed .venv and install deps from pyproject.toml (uv sync)
#   2. Download Swin-B backbone + released PDFNet checkpoint (download_checkpoint.py)
#   3. Clone Depth-Anything-V2 and drop its `depth_anything_v2/` package into DAM_V2/
#   4. Download the Depth-Anything-V2 encoder weights into DAM_V2/checkpoints/
#   5. Run DAM_V2/compute_depth_maps.py on data/interior_segmentation/train.csv
#      (and test.csv if present) to precompute pseudo-depth maps, and fill the
#      `depth_path` column in those CSVs.
#
# Idempotent — re-running skips work that's already done.
#
# README step 1 (the DIS-5K download) is intentionally NOT performed: this fork
# trains on the interior-segmentation datasets already in ./data, not DIS-5K.
#
# Requires `uv` on PATH (https://docs.astral.sh/uv/). The conda flow from the
# README is replaced with `uv venv` + `uv sync` against pyproject.toml.
#
# Env-var toggles:
#   SKIP_ENV=1            skip step 1
#   SKIP_CHECKPOINTS=1    skip steps 2 and 4 (network downloads)
#   SKIP_DAMV2=1          skip step 3 (git clone of Depth-Anything-V2)
#   SKIP_DEPTH=1          skip step 5 (depth-map precomputation)
#   DAMV2_ENCODER=vitb    which DAM-V2 encoder to fetch + use (vits|vitb|vitl)
#
# Usage:
#   ./setup.sh
#   SKIP_ENV=1 ./setup.sh
#   DAMV2_ENCODER=vitl ./setup.sh

set -euo pipefail

cd "$(dirname "$0")"

SKIP_ENV=${SKIP_ENV:-0}
SKIP_CHECKPOINTS=${SKIP_CHECKPOINTS:-0}
SKIP_DAMV2=${SKIP_DAMV2:-0}
SKIP_DEPTH=${SKIP_DEPTH:-0}
DAMV2_ENCODER=${DAMV2_ENCODER:-vitb}
# Where the interior-segmentation datasets + CSVs live (i.e. the directory
# whose subdirs are `kw*_interior_segmentation/` and whose CSVs reference
# image paths relative to it).
DATA_ROOT=${DATA_ROOT:-data/interior_segmentation}

if ! command -v uv >/dev/null; then
    echo "ERROR: 'uv' not found in PATH."
    echo "       Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 2
fi

step() {
    echo
    echo "============================================================"
    echo "[setup] $*"
    echo "============================================================"
}

# ---- 1. uv venv + uv sync ----
if [[ "$SKIP_ENV" != "1" ]]; then
    step "1/5  create .venv (uv) and install deps (uv sync)"
    if [[ ! -d .venv ]]; then
        # Hermetic venv — must NOT use --system-site-packages, otherwise the
        # host's cu13 torch shadows the cu128 wheels pulled from the PyTorch
        # index pinned in pyproject.toml.
        uv venv
    else
        echo ".venv already present"
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    uv sync
else
    step "1/5  skip environment setup (SKIP_ENV=1)"
    # Still activate if it's there, so subsequent steps use the right interpreter.
    if [[ -z "${VIRTUAL_ENV:-}" && -d .venv ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
fi

if ! command -v python >/dev/null; then
    echo "ERROR: 'python' not found after env setup. Inspect .venv/."
    exit 2
fi

# ---- 2. Swin-B + PDFNet checkpoints ----
if [[ "$SKIP_CHECKPOINTS" != "1" ]]; then
    step "2/5  download Swin-B + PDFNet checkpoints"
    SWIN=checkpoints/swin_base_patch4_window12_384_22k.pth
    PDFNET=checkpoints/PDFNet_Best.pth
    if [[ -f "$SWIN" && -f "$PDFNET" ]]; then
        echo "both checkpoints already present, skipping download_checkpoint.py"
    else
        python download_checkpoint.py
    fi
else
    step "2/5  skip checkpoint download (SKIP_CHECKPOINTS=1)"
fi

# ---- 3. clone Depth-Anything-V2 ----
if [[ "$SKIP_DAMV2" != "1" ]]; then
    step "3/5  clone Depth-Anything-V2 into DAM_V2/"
    if [[ -d DAM_V2/depth_anything_v2 ]]; then
        echo "DAM_V2/depth_anything_v2/ already present, skipping clone"
    else
        TMP=$(mktemp -d)
        # shellcheck disable=SC2064
        trap "rm -rf '$TMP'" EXIT
        git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 "$TMP/dav2"
        # We only need the importable package; the rest of the repo (demos, READMEs)
        # would clutter DAM_V2/.
        cp -r "$TMP/dav2/depth_anything_v2" DAM_V2/depth_anything_v2
        rm -rf "$TMP"
        trap - EXIT
        echo "  -> DAM_V2/depth_anything_v2/"
    fi
else
    step "3/5  skip DAM-V2 clone (SKIP_DAMV2=1)"
fi

# ---- 4. Depth-Anything-V2 weights ----
if [[ "$SKIP_CHECKPOINTS" != "1" ]]; then
    step "4/5  download Depth-Anything-V2 ($DAMV2_ENCODER) weights"
    DAMV2_CKPT=DAM_V2/checkpoints/depth_anything_v2_${DAMV2_ENCODER}.pth
    if [[ -f "$DAMV2_CKPT" ]]; then
        echo "$DAMV2_CKPT already present, skipping"
    else
        case "$DAMV2_ENCODER" in
            vits) DAMV2_FAMILY=Small ;;
            vitb) DAMV2_FAMILY=Base ;;
            vitl) DAMV2_FAMILY=Large ;;
            *) echo "Unknown DAMV2_ENCODER=$DAMV2_ENCODER (expected vits|vitb|vitl)"; exit 2 ;;
        esac
        URL="https://huggingface.co/depth-anything/Depth-Anything-V2-${DAMV2_FAMILY}/resolve/main/depth_anything_v2_${DAMV2_ENCODER}.pth?download=true"
        mkdir -p DAM_V2/checkpoints
        echo "  $URL"
        echo "  -> $DAMV2_CKPT"
        if command -v wget >/dev/null; then
            wget --show-progress -O "$DAMV2_CKPT" "$URL"
        elif command -v curl >/dev/null; then
            curl -L --fail -o "$DAMV2_CKPT" "$URL"
        else
            echo "ERROR: neither wget nor curl is installed"
            exit 2
        fi
    fi
else
    step "4/5  skip DAM-V2 weights (SKIP_CHECKPOINTS=1)"
fi

# ---- 5. precompute depth maps for train.csv (and test.csv) ----
if [[ "$SKIP_DEPTH" != "1" ]]; then
    step "5/5  precompute depth maps via DAM_V2/compute_depth_maps.py"
    any=0
    for csv in "$DATA_ROOT/index.csv" "$DATA_ROOT/train.csv" "$DATA_ROOT/test.csv"; do
        if [[ -f "$csv" ]]; then
            any=1
            echo
            echo "--- $csv ---"
            ( cd DAM_V2 && python compute_depth_maps.py \
                --csv "../$csv" \
                --data-root "../$DATA_ROOT" \
                --encoder "$DAMV2_ENCODER" \
                --update-csv )
        fi
    done
    if [[ "$any" == "0" ]]; then
        echo "no CSVs found under $DATA_ROOT. Run build_dataset_csv.py first to create"
        echo "$DATA_ROOT/index.csv, then split it into train.csv and test.csv."
    fi
else
    step "5/5  skip depth-map precomputation (SKIP_DEPTH=1)"
fi

step "setup complete"
