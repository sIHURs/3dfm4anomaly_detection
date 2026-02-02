#!/usr/bin/env bash
set -euo pipefail

source /home/wangyifa/anaconda3/tmp/miniconda3/etc/profile.d/conda.sh
# conda deactivate
# conda activate 3dgs-mcmc-ampere-cuda12.9

# env path
BASE="/home/wangyifa/tmp/3dfm4anomaly_detection"
Factory="/home/wangyifa/tmp/3dfm4anomaly_detection/factory"
unset PYTHONPATH
export PYTHONPATH="$Factory:$Factory/splatpose:$Factory/vggt_low_vram:$Factory/vggtx:$Factory/gaussian_splatting:${PYTHONPATH:-}"

ROOT_DIR="/home/wangyifa/tmp/3dfm4anomaly_detection/scripts/experiment_RAD_nonmask_vggt"
GS_MODEL_DIR="$ROOT_DIR/3dgs_model"
DATA_DIR="/home/wangyifa/tmp/3dfm4anomaly_detection/data/MAD-Sim"
MCMC_CONFIG="factory/gaussian_splatting_mcmc/configs"

SEED=0

LOG_DIR="$ROOT_DIR/3dgs_mcmc_training_initsfm_logs"
PREFIX=""
mkdir -p "$LOG_DIR"


classes=(
    "rubberduck"
    # "binderclip2" 
    # "bowl_upright" "box" "can"
    # "charger" "cup1_upright" "cup2_upright"
    # "gluebottle" "phonecase2" 
    # "spoon_upright" "tennisball"
)


for cls in "${classes[@]}"; do
    SCENE_DIR="$GS_MODEL_DIR/$cls"
    LOG_FILE="$LOG_DIR/${cls}${PREFIX}.log"

    [ -d "$SCENE_DIR" ] || {
        echo "⚠️  Skip $cls: directory not found ($SCENE_DIR)"
        continue
    }

    echo ">>> [`date`] Start processing $cls <<<"

    # echo "> Convert vggt output to 3dgs input format"
    # python convert.py --source_path "$SCENE_DIR" --skip_matching
    
    # echo "> Correct roation center"
    # echo ">>> [`date`] Correct rotation center"
    # python utils/recenter_colmap.py \
    #     --in_sparse_dir "$SCENE_DIR" \
    #     --out_sparse_dir "$SCENE_DIR" \
    #     --overwrite \
    #     --save_T "$SCENE_DIR/recenter_T.npy"

    echo ">>> 3dgs vanilla training: $SCENE_DIR"
    if python train_3dgs.py -s "$SCENE_DIR" --save_iterations $(seq 10000 10000 30000) --test_iterations $(seq 10000 10000 30000)  2>&1 | stdbuf -oL -eL tee "$LOG_FILE"; then
        echo "✅ Done: $SCENE_NAME"
    else
        echo "❌ Failed: $SCENE_NAME (see $LOG_FILE)"
    fi

    echo "========== Done $cls =========="
done