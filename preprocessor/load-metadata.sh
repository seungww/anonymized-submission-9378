#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data-prefix> <base-directory>"
    echo "  <data-prefix>: one of [app, meta]"
    echo "  <base-directory>: e.g., features/meta-free-apps"
    exit 1
fi

DATA_PREFIX=$1   # e.g., app, meta
BASE_DIR=$2      # e.g., features/meta-free-apps

if [ "$DATA_PREFIX" = "all" ]; then
    PTYPE="app-and-meta"
else
    PTYPE="$DATA_PREFIX"
fi

SCRIPT="python loading-metadata.py"

# Feature keys to exclude from processing
EXCLUDED_KEYS=("PKT_index" "PKT_tag" "PKT_len")

# Full feature key list
FEATURE_KEYS=(
    "PKT_dir" "PKT_ts" "PKT_ts_diff" "PKT_ts_rel" "PKT_dirlen"
    "IP_version" "IP_ihl" "IP_tos" "IP_len" "IP_id" "IP_flags" "IP_frag"
    "IP_ttl" "IP_proto" "IP_chksum" "IP_src" "IP_dst"
    "TCP_sport" "TCP_dport" "TCP_seq" "TCP_ack"
    "TCP_dataofs" "TCP_reserved" "TCP_flags" "TCP_window" "TCP_chksum" "TCP_urgptr"
    "UDP_sport" "UDP_dport" "UDP_len" "UDP_chksum"
)

echo "Running all feature-key commands for:"
echo "  DATA_PREFIX: $DATA_PREFIX"
echo "  BASE_DIR:    $BASE_DIR"
echo "  PTYPE used:  $PTYPE"
echo ""

for KEY in "${FEATURE_KEYS[@]}"; do
    if [[ " ${EXCLUDED_KEYS[@]} " =~ " ${KEY} " ]]; then
        echo "Skipping excluded feature: ${KEY}"
        continue
    fi

    PROJECT_NAME="${DATA_PREFIX,,}-${KEY,,}"  # e.g., app-tcp_ack or meta-tcp_ack
    CMD="$SCRIPT --featdir $BASE_DIR --ptype $PTYPE --output $PROJECT_NAME --$KEY"

    echo "Running: $CMD"
    $CMD &
done

wait
echo "All feature-key executions completed."

