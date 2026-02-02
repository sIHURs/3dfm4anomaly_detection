#!/usr/bin/env bash
set -euo pipefail

# env path
BASE="/home/wangyifa/tmp/3dfm4anomaly_detection"
Factory="/home/wangyifa/tmp/3dfm4anomaly_detection/factory"
unset PYTHONPATH
export PYTHONPATH="$Factory:$Factory/splatpose:$Factory/vggt_low_vram:$Factory/vggtx:$Factory/gaussian_splatting_mcmc:${PYTHONPATH:-}"

export ROOT_DIR="/home/wangyifa/tmp/3dfm4anomaly_detection/scripts/experiment_RAD_nonmask_vggt"
export GS_MODEL_DIR="$ROOT_DIR/3dgs_model"
export DATA_DIR="/home/wangyifa/tmp/3dfm4anomaly_detection/data/Anomaly_refine_nonmsk"
export MCMC_CONFIG="factory/gaussian_splatting_mcmc/configs"

SEED=0
CONF=1.1
MAX_POINTS=100000

LOG_DIR="$ROOT_DIR/vggt_logs"
PREFIX=""
mkdir -p "$LOG_DIR"


classes=(
    # "rubberduck"
    "binderclip2" 
    "bowl_upright" "box" "can"
    "charger" "cup1_upright" "cup2_upright"
    "gluebottle" "phonecase2" 
    "spoon_upright" "tennisball"
)


for cls in "${classes[@]}"; do
    SCENE_DIR="$GS_MODEL_DIR/$cls"
    LOG_FILE="$LOG_DIR/${cls}${PREFIX}.log"
    export CLS="$cls"

    [ -d "$SCENE_DIR" ] || {
        echo "⚠️  Skip $cls: directory not found ($SCENE_DIR)"
        continue
    }

    # processing the test images
    eval_dir="$DATA_DIR/$cls/test"
    mapfile -t subfolders < <(find "$eval_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

    echo "Find test cases (${#subfolders[@]}):"
    printf '  %s\n' "${subfolders[@]}"


    echo "[`date`] reverse prepared data for ${cls}"
    bash $ROOT_DIR/prepare_data_prefix.sh restore "${subfolders[@]}"

    echo ">>> [`date`] Start processing $cls <<<"

    # python utils/resize_input_images.py --in_dir "$SCENE_DIR/images" --inplace

    echo "[`date`] prepare data for ${cls}"
    bash $ROOT_DIR/prepare_data_prefix.sh prepare "${subfolders[@]}"

    if python -u demo_colmap_vggt_lowvram.py \
        --scene_dir "$SCENE_DIR" \
        --conf_thres_value $CONF \
        --seed "$SEED" \
        --save_depth \
        --max_points $MAX_POINTS \
        --test_sparse_view \
        --eval_dir "$DATA_DIR/$cls/test" \
        2>&1 | tee "$LOG_FILE"; then

        echo "✅ Done with vggt reconstruction: $cls"
    else
        echo "❌ vggt Failed: $cls (see $LOG_FILE)"
    fi

    echo "[`date`] reverse prepared data for ${cls}"
    bash $ROOT_DIR/prepare_data_prefix.sh restore "${subfolders[@]}"

    ##############################################################
    echo "[INFO] Restructuring scene dir: ${SCENE_DIR}"

    # 1) images -> input
    if [ -d "${SCENE_DIR}/images" ]; then
    if [ -d "${SCENE_DIR}/input" ]; then
        echo "[SKIP] '${SCENE_DIR}/input' already exists, keep it."
    else
        mv "${SCENE_DIR}/images" "${SCENE_DIR}/input"
        echo "[OK] Moved 'images' -> 'input'"
    fi
    else
    echo "[SKIP] No 'images' directory found."
    fi

    # 2) sparse/0 -> distorted/sparse/0
    if [ -d "${SCENE_DIR}/sparse/0" ]; then
    if [ -d "${SCENE_DIR}/distorted/sparse/0" ]; then
        echo "[SKIP] '${SCENE_DIR}/distorted/sparse/0' already exists, keep it."
    else
        mkdir -p "${SCENE_DIR}/distorted/sparse"
        mv "${SCENE_DIR}/sparse/0" "${SCENE_DIR}/distorted/sparse/"
        echo "[OK] Moved 'sparse/0' -> 'distorted/sparse/0'"
    fi
    else
    echo "[SKIP] No 'sparse/0' directory found."
    fi

    # 3) remove old sparse if empty
    if [ -d "${SCENE_DIR}/sparse" ]; then
    if [ -z "$(ls -A "${SCENE_DIR}/sparse")" ]; then
        rm -rf "${SCENE_DIR}/sparse"
        echo "[OK] Removed empty 'sparse' directory"
    else
        echo "[WARN] 'sparse' not empty, keeping it: $(ls "${SCENE_DIR}/sparse")"
    fi
    else
    echo "[SKIP] No 'sparse' directory found."
    fi

    echo "[DONE] Folder structure updated."

    #################################################################

    echo ">>> [`date`] Convert vggt output to 3dgs input format"
    python convert.py --source_path "$SCENE_DIR" --skip_matching

    echo ">>> [`date`] Correct rotation center"
    python utils/recenter_colmap.py \
        --in_sparse_dir "$SCENE_DIR" \
        --out_sparse_dir "$SCENE_DIR" \
        --overwrite \
        --save_T "$SCENE_DIR/recenter_T.npy"

    echo ">>> [`date`] Correct rotation center - colmap"
    python utils/recenter_transforms.py \
        --T_path "$SCENE_DIR/recenter_T.npy" \
        --transforms_in  "$SCENE_DIR/transforms_anomaly_free_poses_uncentered.json" \
        --transforms_out "$SCENE_DIR/transforms_anomaly_free_poses.json" \
        --matrix_type c2w


    echo ">>> [`date`] Correct rotation center - transforms.json"
    python utils/recenter_transforms.py \
        --T_path "$SCENE_DIR/recenter_T.npy" \
        --transforms_in  "$SCENE_DIR/transforms_query_poses_uncentered.json" \
        --transforms_out "$SCENE_DIR/transforms_query_poses.json" \
        --matrix_type c2w

    echo "========== Done $cls =========="
done