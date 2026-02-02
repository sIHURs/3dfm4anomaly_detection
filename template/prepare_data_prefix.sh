#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

MODE="${1:-}"
shift || true
if [[ "$MODE" != "prepare" && "$MODE" != "restore" ]]; then
  echo "Usage:"
  echo "  $0 {prepare|restore} [class1 class2 ...]"
  echo "  $0 {prepare|restore} class1,class2,..."
  echo ""
  echo "Or via env:"
  echo "  CLASSES=\"good,Stains\" $0 {prepare|restore}"
  exit 1
fi

# -----------------------
# USER CONFIG
# -----------------------

# If you don't need DIR_B filtering, set ENABLE_FILTER_B=0
ENABLE_FILTER_B="${ENABLE_FILTER_B:-1}"

GS_MODEL_DIR="${GS_MODEL_DIR:-}"   # e.g. /path/to/3dgs_model
DATA_DIR="${DATA_DIR:-}"           # e.g. /path/to/MAD-Sim
CLS="${CLS:-}"                     # e.g. 01Gorilla

TEST_DIR="${DATA_DIR}/${CLS}/test"
GT_DIR="${DATA_DIR}/${CLS}/ground_truth"

# -----------------------
# CLASSES: from args or env
# -----------------------
# Priority: CLI args > env CLASSES > default
CLASSES_ARR=()

if (( $# > 0 )); then
  # If user passed a single "a,b,c" token, split by comma.
  if (( $# == 1 )) && [[ "$1" == *","* ]]; then
    IFS=',' read -r -a CLASSES_ARR <<< "$1"
  else
    CLASSES_ARR=("$@")
  fi
elif [[ -n "${CLASSES:-}" ]]; then
  # Env var: "a,b,c" or "a b c"
  if [[ "$CLASSES" == *","* ]]; then
    IFS=',' read -r -a CLASSES_ARR <<< "$CLASSES"
  else
    read -r -a CLASSES_ARR <<< "$CLASSES"
  fi
else
  CLASSES_ARR=("good" "Stains")
fi

# Trim empty entries (defensive)
_tmp=()
for c in "${CLASSES_ARR[@]}"; do
  [[ -n "$c" ]] && _tmp+=("$c")
done
CLASSES_ARR=("${_tmp[@]}")

echo "$TEST_DIR"
echo "$GT_DIR"
echo "[INFO] CLASSES = ${CLASSES_ARR[*]}"

# Supported extensions
EXTS=("png" "jpg" "jpeg")

# -----------------------
# helpers
# -----------------------
ts() { date "+%Y-%m-%d %H:%M:%S"; }

ensure_dir() {
  local d="$1"
  [[ -d "$d" ]] || { echo "[ERROR] Missing dir: $d"; exit 1; }
}

# -----------------------
# MODE: prepare
# -----------------------
if [[ "$MODE" == "prepare" ]]; then
  echo "[$(ts)] Prepare the data"

  # ---- 1) rename test/ & ground_truth/ with prefix, using tmp + symlink ----
  for cls in "${CLASSES_ARR[@]}"; do
    TEST_CLS_DIR="$TEST_DIR/$cls"
    GT_CLS_DIR="$GT_DIR/$cls"

    ensure_dir "$TEST_CLS_DIR"

    TMP_TEST="$TEST_CLS_DIR/__tmp_backup__"
    mkdir -p "$TMP_TEST"

    echo "[INFO] Prefixing test/$cls ..."

    # test: handle png/jpg/jpeg
    for ext in "${EXTS[@]}"; do
      for f in "$TEST_CLS_DIR"/*."$ext"; do
        [[ -f "$f" ]] || continue
        base="$(basename "$f")"

        # Skip already prefixed
        if [[ "$base" == query${cls}_* ]]; then
          continue
        fi

        stem="${base%.*}"
        ext2=".${base##*.}"  # keep original ext
        new_name="query${cls}_${stem}${ext2}"

        mv "$f" "$TMP_TEST/$base"
        ln -sf "$TMP_TEST/$base" "$TEST_CLS_DIR/$new_name"
      done
    done

    # ground_truth: only if dir exists (good might not exist)
    if [[ -d "$GT_CLS_DIR" ]]; then
      TMP_GT="$GT_CLS_DIR/__tmp_backup__"
      mkdir -p "$TMP_GT"

      echo "[INFO] Prefixing ground_truth/$cls ..."

      for ext in "${EXTS[@]}"; do
        for f in "$GT_CLS_DIR"/*."$ext"; do
          [[ -f "$f" ]] || continue
          base="$(basename "$f")"

          # 跳过已经是 query 前缀的（两种：带_mask / 不带_mask）
          if [[ "$base" == query${cls}_* ]]; then
            continue
          fi

          ext2=".${base##*.}"

          if [[ "$base" == *_mask."$ext" ]]; then
            # 原始是 xxx_mask.png
            stem="${base%_mask.*}"   # xxx
            new_name="query${cls}_${stem}_mask${ext2}"
          else
            # 原始是 xxx.png
            stem="${base%.*}"        # xxx
            new_name="query${cls}_${stem}${ext2}"
          fi

          mv "$f" "$TMP_GT/$base"
          ln -sf "$TMP_GT/$base" "$GT_CLS_DIR/$new_name"
        done
      done
    else
      echo "[INFO] Skip ground_truth/$cls (dir not found)"
    fi
  done

  # ---- 2) integrate your DIR_A -> filter DIR_B logic (optional) ----
  if [[ "$ENABLE_FILTER_B" -eq 1 ]]; then
    echo "[INFO] Filter B by A for class: $CLS"

    DIR_A="$GS_MODEL_DIR/$CLS/images"      # contains train_000.png/jpg
    DIR_B="$DATA_DIR/$CLS/train/good"      # contains 000.png/jpg
    ensure_dir "$DIR_A"
    ensure_dir "$DIR_B"

    TMP_DIR="${DIR_B}/__tmp_backup__"
    mkdir -p "$TMP_DIR"

    echo "[INFO] Collecting indices from A..."
    declare -A keep_ids=()

    # collect ids from A, support multiple extensions
    for ext in "${EXTS[@]}"; do
      for f in "$DIR_A"/train_*."$ext"; do
        [[ -f "$f" ]] || continue
        base="$(basename "$f")"      # train_000.png
        id="${base#train_}"          # 000.png
        id="${id%.*}"                # 000
        keep_ids["$id"]=1
      done
    done

    echo "[INFO] Filtering B..."
    for ext in "${EXTS[@]}"; do
      for f in "$DIR_B"/*."$ext"; do
        [[ -f "$f" ]] || continue
        base="$(basename "$f")"      # 000.png
        id="${base%.*}"              # 000

        if [[ -z "${keep_ids[$id]:-}" ]]; then
          mv "$f" "$TMP_DIR/"
        fi
      done
    done
  fi

  echo "[$(ts)] DONE: prepare"
  exit 0
fi

# -----------------------
# MODE: restore
# -----------------------
if [[ "$MODE" == "restore" ]]; then
  echo "[$(ts)] Restore the data"

  # ---- 1) restore test/ & ground_truth/ ----
  for cls in "${CLASSES_ARR[@]}"; do
    TEST_CLS_DIR="$TEST_DIR/$cls"
    GT_CLS_DIR="$GT_DIR/$cls"

    # --- restore test if exists ---
    if [[ -d "$TEST_CLS_DIR" ]]; then
      TMP_TEST="$TEST_CLS_DIR/__tmp_backup__"
      echo "[INFO] Restoring test/$cls ..."

      # remove symlinks
      for ext in "${EXTS[@]}"; do
        rm -f "$TEST_CLS_DIR"/query${cls}_*."$ext" 2>/dev/null || true
      done

      # move originals back
      if [[ -d "$TMP_TEST" ]]; then
        for ext in "${EXTS[@]}"; do
          mv "$TMP_TEST"/*."$ext" "$TEST_CLS_DIR/" 2>/dev/null || true
        done
        rmdir "$TMP_TEST" 2>/dev/null || true
      fi
    else
      echo "[INFO] Skip restore test/$cls (dir not found)"
    fi

    # --- restore ground_truth if exists ---
    if [[ -d "$GT_CLS_DIR" ]]; then
      TMP_GT="$GT_CLS_DIR/__tmp_backup__"
      echo "[INFO] Restoring ground_truth/$cls ..."

      # 1) remove symlinks (both *_mask and non-mask)
      for ext in "${EXTS[@]}"; do
        rm -f "$GT_CLS_DIR"/query${cls}_*."$ext" 2>/dev/null || true
      done

      # 2) move originals back (both *_mask and non-mask)
      if [[ -d "$TMP_GT" ]]; then
        for ext in "${EXTS[@]}"; do
          mv "$TMP_GT"/*."$ext" "$GT_CLS_DIR/" 2>/dev/null || true
        done
        rmdir "$TMP_GT" 2>/dev/null || true
      fi
    else
      echo "[INFO] Skip restore ground_truth/$cls (dir not found)"
    fi
  done

  # ---- 2) restore filtered files back to DIR_B ----
  if [[ "$ENABLE_FILTER_B" -eq 1 ]]; then
    DIR_B="$DATA_DIR/$CLS/train/good"
    if [[ -d "$DIR_B" ]]; then
      TMP_DIR="${DIR_B}/__tmp_backup__"
      if [[ -d "$TMP_DIR" ]]; then
        echo "[INFO] Restoring filtered files back to: $DIR_B"
        for ext in "${EXTS[@]}"; do
          mv "$TMP_DIR"/*."$ext" "$DIR_B/" 2>/dev/null || true
        done
        rmdir "$TMP_DIR" 2>/dev/null || true
      fi
    fi
  fi

  echo "[$(ts)] DONE: restore"
  exit 0
fi
