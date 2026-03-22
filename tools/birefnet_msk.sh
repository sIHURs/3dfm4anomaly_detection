#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: $0 /path/to/root /path/to/root_birefnet /path/to/run_birefnet_mask_whitebg.py}"
ROOT_OUT="${2:?usage: $0 /path/to/root /path/to/root_birefnet /path/to/run_birefnet_mask_whitebg.py}"
PY_SCRIPT="${3:?usage: $0 /path/to/root /path/to/root_birefnet /path/to/run_birefnet_mask_whitebg.py}"

DEVICE="${DEVICE:-cuda}"        # cuda|cpu|auto
INFER_SIZE="${INFER_SIZE:-1024}"
THRESH="${THRESH:-128}"
OVERWRITE="${OVERWRITE:-1}"     # 覆盖输出（建议=1）

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

echo "[info] ROOT      : $ROOT"
echo "[info] ROOT_OUT  : $ROOT_OUT"
echo "[info] PY_SCRIPT : $PY_SCRIPT"
echo "[info] device=$DEVICE infer_size=$INFER_SIZE thresh=$THRESH overwrite=$OVERWRITE"

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

  # 1) 同步该 class（保持结构/命名一致）
  mkdir -p "$dst_cls"
  rsync -a "$src_cls/" "$dst_cls/"

  # 2) 只处理 train/test（排除 ground_truth），并按“包含 png 的目录”运行一次 python
  find "$dst_cls" -type f -name "*.png" \
    ! -path "*/ground_truth/*" \
    ! -path "*/train_msk/*" \
    ! -path "*/test_msk/*" \
    -printf '%h\n' | sort -u | while read -r dir; do

      # 跳过 masks 目录（我们会临时用到，但马上搬走）
      if [[ "$dir" == *"/masks" ]]; then
        continue
      fi

      # 只处理 train/ 与 test/ 下的目录（更严格）
      rel="${dir#"$dst_cls"/}"
      if [[ "$rel" != train/* && "$rel" != test/* ]]; then
        continue
      fi

      echo "[info]  folder: $dir"

      # 2.1 在当前 dir 覆盖输出白底抠图；mask 临时输出到 dir/masks
      python "$PY_SCRIPT" \
        --input_dir "$dir" \
        --pattern "*.png" \
        --masked_subdir "." \
        --masks_subdir "masks" \
        --device "$DEVICE" \
        --infer_size "$INFER_SIZE" \
        --threshold "$THRESH" \
        $OW_FLAG

      # 2.2 把 mask 搬到外层：train_msk / test_msk
      if [[ -d "$dir/masks" ]]; then
        if [[ "$rel" == train/* ]]; then
          sub="${rel#train/}"         # e.g. good 或 good/some
          target="$dst_cls/train_msk/$sub"
        else
          sub="${rel#test/}"          # e.g. scratched 或 scratched/xxx
          target="$dst_cls/test_msk/$sub"
        fi

        mkdir -p "$target"
        # mask 文件名与原图一致
        mv "$dir/masks/"*.png "$target/" 2>/dev/null || true
        rmdir "$dir/masks" 2>/dev/null || true
      fi
    done

  echo "[info] done class: $cls"
done

echo "[done] all classes processed."