#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: $0 /path/to/root /path/to/root_out /path/to/run_birefnet_mask_whitebg_refine.py}"
ROOT_OUT="${2:?usage: $0 /path/to/root /path/to/root_out /path/to/run_birefnet_mask_whitebg_refine.py}"
PY_SCRIPT="${3:?usage: $0 /path/to/root /path/to/root_out /path/to/run_birefnet_mask_whitebg_refine.py}"

DEVICE="${DEVICE:-cuda}"        # cuda|cpu|auto
INFER_SIZE="${INFER_SIZE:-1024}"
THRESH="${THRESH:-128}"
OVERWRITE="${OVERWRITE:-1}"     # 1=覆盖输出
FP16="${FP16:-1}"               # 1=开启 --fp16（仅cuda有效）
KEEP_SOFT="${KEEP_SOFT:-0}"     # 1=保留 soft mask；0=删除
REFINE="${REFINE:-1}"           # 1=开启 refine_foreground；0=关闭
REFINE_R="${REFINE_R:-90}"
FILL_HOLES="${FILL_HOLES:-1}"   # 1=填洞；0=不填
MASKED_SUBDIR="${MASKED_SUBDIR:-.}"  # "." 表示覆盖原图；或 masked_white

# 你可以在这里列 class；也可以改成自动扫描 ROOT 下所有目录
classes=(
    # "rubberduck"
    # "binderclip2"
    # "bowl_upright"
    # "box"
    # "can"
    "charger"
    # "cup1_upright"
    # "cup2_upright"
    # "gluebottle"
    # "spoon_upright"
    # "tennisball"
    # "phonecase2"

    # "binderclip"  # 3dgs mcmc traninig is wrong
    # "cup2_upright2"
    # "cup2_upright3"
    # "phonecase"
    # "gluebottle2"
    # "spraybottle2" # also
)

OW_FLAG=""
if [[ "${OVERWRITE}" == "1" ]]; then
  OW_FLAG="--overwrite"
fi

FP16_FLAG=""
if [[ "${FP16}" == "1" ]]; then
  FP16_FLAG="--fp16"
fi

REFINE_FLAG=""
if [[ "${REFINE}" == "1" ]]; then
  REFINE_FLAG="--refine_foreground --refine_r ${REFINE_R}"
fi

FILL_FLAG=""
if [[ "${FILL_HOLES}" == "1" ]]; then
  FILL_FLAG="--fill_holes"
fi

SOFT_FLAG="--save_soft --soft_subdir masks_soft"

echo "[info] ROOT      : $ROOT"
echo "[info] ROOT_OUT  : $ROOT_OUT"
echo "[info] PY_SCRIPT : $PY_SCRIPT"
echo "[info] device=$DEVICE infer_size=$INFER_SIZE thresh=$THRESH overwrite=$OVERWRITE fp16=$FP16 keep_soft=$KEEP_SOFT"
echo "[info] masked_subdir=$MASKED_SUBDIR refine=$REFINE refine_r=$REFINE_R fill_holes=$FILL_HOLES"

mkdir -p "$ROOT_OUT"

for cls in "${classes[@]}"; do
  src_cls="$ROOT/$cls"
  dst_cls="$ROOT_OUT/$cls"

  echo "============================================================"
  echo "[info] class: $cls"

  if [[ ! -d "$src_cls" ]]; then
    echo "[warn] skip (missing): $src_cls"
    continue
  fi

  mkdir -p "$dst_cls"

  # 同步时排除已有mask与ground_truth，避免旧结果带过去
  rsync -a \
    --exclude "ground_truth/" \
    --exclude "train_msk/" \
    --exclude "test_msk/" \
    "$src_cls/" "$dst_cls/"

  # 找到 train/test 下包含 png 的目录
  find "$dst_cls" -type f -name "*.png" \
    ! -path "*/ground_truth/*" \
    ! -path "*/train_msk/*" \
    ! -path "*/test_msk/*" \
    -printf '%h\n' | sort -u | while read -r dir; do

      # 跳过脚本生成的中间目录（避免重复处理）
      if [[ "$dir" == *"/masks_soft"* || "$dir" == *"/masks"* ]]; then
        continue
      fi

      rel="${dir#"$dst_cls"/}"
      if [[ "$rel" != train/* && "$rel" != test/* ]]; then
        continue
      fi

      # 只处理当前目录下 png（避免递归导致重复处理）
      if ! find "$dir" -maxdepth 1 -type f -name "*.png" -print -quit | grep -q .; then
        continue
      fi

      echo "[info]  folder: $dir"

      # 运行 python：生成 white masked + binary masks + (optional) soft masks
      python "$PY_SCRIPT" \
        --input_dir "$dir" \
        --pattern "*.png" \
        --infer_size "$INFER_SIZE" \
        --threshold "$THRESH" \
        --masked_subdir "$MASKED_SUBDIR" \
        --masks_subdir "masks_bin" \
        --device "$DEVICE" \
        $FP16_FLAG \
        $OW_FLAG \
        $FILL_FLAG \
        $REFINE_FLAG \
        $SOFT_FLAG

      # 把 binary mask 搬到外层：train_msk / test_msk
      if [[ -d "$dir/masks_bin" ]]; then
        if [[ "$rel" == train/* ]]; then
          sub="${rel#train/}"
          target="$dst_cls/train_msk/$sub"
        else
          sub="${rel#test/}"
          target="$dst_cls/test_msk/$sub"
        fi

        mkdir -p "$target"
        find "$dir/masks_bin" -maxdepth 1 -type f -name "*.png" -exec mv -f {} "$target/" \;
        rmdir "$dir/masks_bin" 2>/dev/null || true
      fi

      # soft mask 是否保留
      if [[ "${KEEP_SOFT}" != "1" ]]; then
        rm -rf "$dir/masks_soft" 2>/dev/null || true
      fi
    done

  echo "[info] done class: $cls"
done

echo "[done] all classes processed."