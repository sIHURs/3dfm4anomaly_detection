#!/usr/bin/env bash
set -euo pipefail

# env path
BASE="/home/wangyifa/tmp/3dfm4anomaly_detection"
Factory="/home/wangyifa/tmp/3dfm4anomaly_detection/factory"
unset PYTHONPATH
export PYTHONPATH="$Factory:$Factory/splatpose:$Factory/gaussian_splatting:$Factory/vggt_low_vram:$Factory/vggtx:${PYTHONPATH:-}"

UTILS_DIR="/home/wangyifa/tmp/3dfm4anomaly_detection/utils"
DATA_DIR="/home/wangyifa/tmp/3dfm4anomaly_detection/data/PIAD_dataset/Real/CL"
ROOT_DIR="/home/wangyifa/tmp/3dfm4anomaly_detection/scripts/experiment_PIAD_real_colmap_3dgsmcmc"
GS_MODEL_DIR="$ROOT_DIR"/3dgs_model
RES_DIR="$ROOT_DIR"/ad_result

LOG_DIR="$ROOT_DIR/ad_eval_logs"
mkdir -p "$LOG_DIR"
mkdir -p "$RES_DIR"

classes=(
   "01Valve"
)


# classes=(
#    "01Valve" "02Tube" "03Cup" "04USB" "05Joint"
#    "06PaperCup" "07Lighter" "08Cube" "09Lamp" "10Bolt"
# )

for cls in "${classes[@]}"; do
    echo "========== Processing $cls =========="

    log_file="${LOG_DIR}/${cls}_k500_seed0.log"
    
    {
        # echo "[`date`] Start Optimize pcd and gaussians for ${cls}"
        # Class_DIR="$GS_MODEL_DIR"/"$cls"

        # python utils/optimize_3dgs_pcd_V2.py \
        #     --source_path "$Class_DIR" \
        #     --model_path "" \
        #     --sparse_dir "$Class_DIR/sparse/0" \
        #     --image_dir "$Class_DIR/images" \
        #     --model_iteration 30000 \
        #     --thresholds 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 \
        #     --csv_out "$Class_DIR/optimize_result/threshold_sweep.csv" \
        #     --device cuda \
        #     --export_filtered \
        #     --analyze_ratio

        # python utils/optimize_3dgs_gaussians.py \
        #     --ply_path "$Class_DIR/output/point_cloud/iteration_30000/point_cloud_clean_t0.100.ply" \
        #     --output_ply "$Class_DIR/output/point_cloud/iteration_30000/point_cloud_clean_t0.100_gaussiansOpt.ply" \
        #     --size_percentile 96 \
        #     --plot_dir "$Class_DIR/output/point_cloud/iteration_30000/gaussians_analysis"

        echo "[`date`] Start demo_splatpose_pipeline for ${cls}"

        python demo_splatpose_pipeline.py \
            --model_path_splatpose "$GS_MODEL_DIR" \
            --data_path "$DATA_DIR" \
            --result "$RES_DIR" \
            --classname "$cls" \
            --prefix "k500_seed0"\
            --json_name "transforms.json" \
            --pcd_name "point_cloud.ply" \
            --seed 0 \
            -v 1 \
            -k 500

        echo "[`date`] Finished ${cls}"
    } 2>&1 | tee "$log_file"

    echo "========== Done $cls, log: $log_file =========="
done


