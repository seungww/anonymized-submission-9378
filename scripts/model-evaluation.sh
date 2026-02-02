#!/bin/bash

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <data-prefix> <base-directory> <model> [--gpu1]"
    echo "  <data-prefix>: one of [app, meta]"
    echo "  <base-directory>: e.g., projects"
    echo "  [--gpu1]: (optional) run on GPU 1"
    exit 1
fi

DATA_PREFIX=$1
BASE_DIR=$2
MODEL=$3

# Check optional GPU flag
USE_GPU1=false
if [ "$#" -eq 3 ] && [ "$3" == "--gpu1" ]; then
    USE_GPU1=true
fi

if [[ "$DATA_PREFIX" != "app" && "$DATA_PREFIX" != "meta" ]]; then
    echo "Error: data-prefix must be one of: app, meta"
    exit 1
fi

# Metadata list
metadata_names=(
    "ip_len"
    #"pkt_dirlen" "pkt_dir" "pkt_ts_rel" "pkt_ts_diff"
    #"ip_src" "ip_dst" "tcp_sport" "tcp_dport"
    #"ip_version" "ip_ihl" "ip_ttl" "ip_tos" "ip_proto" "ip_len" "ip_id" "ip_frag" "ip_flags" "ip_chksum"
    #"tcp_window" "tcp_urgptr" "tcp_seq" "tcp_ack" "tcp_reserved" "tcp_flags" "tcp_dataofs" "tcp_chksum"
)

model_names=(
    "cnn" "rnn" "birnn" "lstm" "bilstm" "gru" "bigru" "transformer"
)

STEP="model_evaluation"
SCRIPT="vrscanner.py"
OUTPUT_DIR="output/${BASE_DIR}/${STEP}"
TS=$(date +%s)
LOG_FILE="${OUTPUT_DIR}/${STEP}_${DATA_PREFIX}_script_${TS}.log"
mkdir -p "$OUTPUT_DIR"

echo "=== Step4. Script Execution Started at $(date) ===" | tee "$LOG_FILE"
echo "Base directory: $BASE_DIR" | tee -a "$LOG_FILE"
echo "Data prefix: $DATA_PREFIX" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a $LOG_FILE

FEATURE="ip_len"
NORM="token"
DATA_PATH="$BASE_DIR/${DATA_PREFIX}-${FEATURE}/${DATA_PREFIX}-${FEATURE}.csv"

KFOLD=5
LR=0.001
#LR=0.0001

TS=$(date +%s)
DEBUG_PATH="${OUTPUT_DIR}/${STEP}_${DATA_PREFIX}_${TS}.debug"
LEADERBOARD_PATH="${OUTPUT_DIR}/${STEP}_${DATA_PREFIX}_${TS}.csv"

# Build full command
CMD="python $SCRIPT --train --path $DATA_PATH --debug_path $DEBUG_PATH --leaderboard_path $LEADERBOARD_PATH --norm $NORM --model $MODEL --kfold $KFOLD --lr $LR --strict"

if [ "$USE_GPU1" = true ]; then
    CMD="CUDA_VISIBLE_DEVICES=1 $CMD"
fi

echo "Running $CMD" | tee -a "$LOG_FILE"
eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
echo "=== Step4. Script Execution Finished at $(date) ===" | tee -a "$LOG_FILE"
