#!/bin/bash

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <data-prefix> <base-directory> [--gpu1]"
    echo "  <data-prefix>: one of [app, meta]"
    echo "  <base-directory>: e.g., projects"
    echo "  [--gpu1]: (optional) run on GPU 1"
    exit 1
fi

DATA_PREFIX=$1
BASE_DIR=$2

# Check optional GPU flag
USE_GPU1=false
if [ "$#" -eq 3 ] && [ "$3" == "--gpu1" ]; then
    USE_GPU1=true
fi

# Validate data prefix
if [[ "$DATA_PREFIX" != "app" && "$DATA_PREFIX" != "meta" ]]; then
    echo "Error: data-prefix must be one of: app, meta"
    exit 1
fi

# Metadata list
metadata_names=(
    "pkt_dirlen" "pkt_dir" "pkt_ts_rel" "pkt_ts_diff"
    "ip_version" "ip_ihl" "ip_ttl" "ip_tos" "ip_src" "ip_dst"
    "ip_proto" "ip_len" "ip_id" "ip_frag" "ip_flags" "ip_chksum"
    "tcp_window" "tcp_urgptr" "tcp_sport" "tcp_dport"
    "tcp_seq" "tcp_ack" "tcp_reserved" "tcp_flags"
    "tcp_dataofs" "tcp_chksum"
)


STEP1_DIR="output/${BASE_DIR}/step1"

STEP="step2"
SCRIPT="vrscanner.py"
OUTPUT_DIR="output/${BASE_DIR}/${STEP}"
LOG_FILE="${OUTPUT_DIR}/${STEP}_${DATA_PREFIX}_script.log"
mkdir -p "$OUTPUT_DIR"


echo "=== Step2. Script Execution Started at $(date) ===" | tee "$LOG_FILE"
echo "Base directory: $BASE_DIR" | tee -a "$LOG_FILE"
echo "Data prefix: $DATA_PREFIX" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a $LOG_FILE

PKTCOUNT=500
EPOCH=5
BATCH_SIZE=32
INPUT_DIM=128
HIDDEN_SIZE=128
FUSION_DIM=128

for NAME in "${metadata_names[@]}"; do
    CSV_PATH="${STEP1_DIR}/step1_${DATA_PREFIX}_${NAME}.csv"

    if [[ ! -f "$CSV_PATH" ]]; then
        echo "[WARN] Skipping missing CSV file: $CSV_PATH" | tee -a "$LOG_FILE"
        continue
    fi

    if [[ "$NAME" == "ip_src" || "$NAME" == "ip_dst" || "$NAME" == "ip_flags" || "$NAME" == "tcp_flags" ]]; then
        BEST_NORM="token"
    else
        BEST_NORM=$(awk -F',' 'NR > 1 { 
            split($1, arr, "_"); 
            if ($3 > max) { max = $3; best = arr[2] } 
        } END { print best }' "$CSV_PATH")
    fi

    echo "[INFO] $NAME → Best normalization: $BEST_NORM" | tee -a "$LOG_FILE"
    DATA_PATH="$BASE_DIR/${DATA_PREFIX}-${NAME}/${DATA_PREFIX}-${NAME}.csv"
    DEBUG_PATH="${OUTPUT_DIR}/${STEP}_${DATA_PREFIX}_${NAME}.debug"
    LEADERBOARD_PATH="${OUTPUT_DIR}/${STEP}_${DATA_PREFIX}_${NAME}.csv"

    CMD="python $SCRIPT --$STEP --path $DATA_PATH --norm $BEST_NORM --debug_path $DEBUG_PATH --leaderboard_path $LEADERBOARD_PATH --pktcount $PKTCOUNT --epoch $EPOCH --batch_size $BATCH_SIZE --input_dim $INPUT_DIM --hidden_size $HIDDEN_SIZE --fusion_dim $FUSION_DIM"

    if [ "$USE_GPU1" = true ]; then
	CMD="CUDA_VISIBLE_DEVICES=1 $CMD"
    fi

    echo "Running: $CMD" | tee -a "$LOG_FILE"
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
done

echo "=== Step2. Script Execution Finished at $(date) ===" | tee -a "$LOG_FILE"

