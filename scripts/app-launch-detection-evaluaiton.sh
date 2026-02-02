#!/bin/bash

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <data-prefix> <base-directory> <unknown-base-directory> [--gpu1]"
    echo "  <data-prefix>: one of [app, meta]"
    echo "  <base-directory>: e.g., projects"
    echo "  <unknown-base-directory>: e.g., app-not-launched"
    echo "  [--gpu1]: (optional) run on GPU 1"
    exit 1
fi

DATA_PREFIX=$1
BASE_DIR=$2
UNKNOWN_BASE_DIR=$3

USE_GPU1=false
if [ "$#" -eq 4 ] && [ "$4" == "--gpu1" ]; then
    USE_GPU1=true
fi

if [[ "$DATA_PREFIX" != "app" && "$DATA_PREFIX" != "meta" ]]; then
    echo "Error: data-prefix must be one of: app, meta"
    exit 1
fi

metadata_names=(
    "ip_len"
    #"pkt_dirlen" "pkt_dir" "pkt_ts_rel" "pkt_ts_diff"
    #"ip_src" "ip_dst" "tcp_sport" "tcp_dport"
    #"ip_version" "ip_ihl" "ip_ttl" "ip_tos" "ip_proto" "ip_len" "ip_id" "ip_frag" "ip_flags" "ip_chksum"
    #"tcp_window" "tcp_urgptr" "tcp_seq" "tcp_ack" "tcp_reserved" "tcp_flags" "tcp_dataofs" "tcp_chksum"
)

STEP2_DIR="output/${BASE_DIR}/step2"

STEP="app_launch_detection_evaluation"
SCRIPT="vrscanner.py"
OUTPUT_DIR="output/${BASE_DIR}/${STEP}"
TS=$(date +%s)
LOG_FILE="${OUTPUT_DIR}/${STEP}_${DATA_PREFIX}_script_${TS}.log"
mkdir -p "$OUTPUT_DIR"

echo "=== App Launch Detection Evaluation. Script Execution Started at $(date) ===" | tee "$LOG_FILE"
echo "Base directory: $BASE_DIR" | tee -a "$LOG_FILE"
echo "Unknown base directory: $UNKNOWN_BASE_DIR" | tee -a "$LOG_FILE"
echo "Data prefix: $DATA_PREFIX" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"

PATH_ARGS=()
UNKNOWN_PATH_ARGS=()
NORM_ARGS=()
MODEL_ARGS=()

for NAME in "${metadata_names[@]}"; do
    CSV_PATH="${STEP2_DIR}/step2_${DATA_PREFIX}_${NAME}.csv"

    if [[ ! -f "$CSV_PATH" ]]; then
        echo "[WARN] Missing CSV: $CSV_PATH" | tee -a "$LOG_FILE"
        continue
    fi

    BEST_ROW=$(awk -F',' '
        NR==1 { for (i=1;i<=NF;i++) header[i]=$i }
        NR>1 {
            for (i=1;i<=NF;i++) row[header[i]]=$i
            if (row["f1-score"] > max) {
                max = row["f1-score"]
                best_name = row["name"]
            }
        }
        END { print best_name }
    ' "$CSV_PATH")

    if [[ -z "$BEST_ROW" ]]; then
        echo "[WARN] Skipping $NAME due to no best row found" | tee -a "$LOG_FILE"
        continue
    fi

    IFS='_' read -ra PARTS <<< "$BEST_ROW"
    BEST_NORM="${PARTS[1]}"
    BEST_MODEL="${PARTS[2]}"

    echo "[INFO] $NAME → norm: $BEST_NORM, model: $BEST_MODEL" | tee -a "$LOG_FILE"

    DATA_PATH="$BASE_DIR/${DATA_PREFIX}-${NAME}/${DATA_PREFIX}-${NAME}.csv"
    UNKNOWN_PATH="$UNKNOWN_BASE_DIR/${DATA_PREFIX}-${NAME}/${DATA_PREFIX}-${NAME}.csv"
    PATH_ARGS+=("$DATA_PATH")
    UNKNOWN_PATH_ARGS+=("$UNKNOWN_PATH")
    NORM_ARGS+=("$BEST_NORM")
    MODEL_ARGS+=("$BEST_MODEL")
done

PKTCOUNT=1000
INPUT_DIM=512
HIDDEN_SIZE=256
NUM_LAYERS=3
DROPOUT=0.3
FUSION_DIM=256
FC_HIDDEN_SIZE=128
EPOCH=50
LR=0.001

TS=$(date +%s)
DEBUG_PATH="${OUTPUT_DIR}/${STEP}_${DATA_PREFIX}_${TS}.debug"
LEADERBOARD_PATH="${OUTPUT_DIR}/${STEP}_${DATA_PREFIX}_${TS}.csv"

for value in $NUM_LAYERS; do
    TS=$(date +%s)
    DEBUG_PATH="${OUTPUT_DIR}/${STEP}_${DATA_PREFIX}_${TS}.debug"
    LEADERBOARD_PATH="${OUTPUT_DIR}/${STEP}_${DATA_PREFIX}_${TS}.csv"
    CMD="python $SCRIPT --$STEP --debug_path $DEBUG_PATH --leaderboard_path $LEADERBOARD_PATH --pktcount $PKTCOUNT --input_dim $INPUT_DIM --hidden_size $HIDDEN_SIZE --num_layers $value --dropout $DROPOUT --fusion_dim $FUSION_DIM --fc_hidden_size $FC_HIDDEN_SIZE --epoch $EPOCH --lr $LR --strict"
    CMD="$CMD --path"
    for path in "${PATH_ARGS[@]}"; do
        CMD="$CMD $path"
    done
    CMD="$CMD --unknown_path"
    for path in "${UNKNOWN_PATH_ARGS[@]}"; do
        CMD="$CMD $path"
    done
    CMD="$CMD --norm"
    for norm in "${NORM_ARGS[@]}"; do
        CMD="$CMD $norm"
    done
    CMD="$CMD --model"
    for model in "${MODEL_ARGS[@]}"; do
        CMD="$CMD $model"
    done

    if [ "$USE_GPU1" = true ]; then
        CMD="CUDA_VISIBLE_DEVICES=1 $CMD"
    fi

    echo "Running $CMD" | tee -a "$LOG_FILE"
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
done
echo "=== App Launch Detection Evaluation. Script Execution Finished at $(date) ===" | tee -a "$LOG_FILE"
