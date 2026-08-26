#!/usr/bin/env bash
# $1 = tag (opus|sonnet)  $2 = model  $3 = effort
set -u
TILE=data/incoming/sinai/tile_cropped_x801_y846_z0.5.tif
TAG=$1; MODEL=$2; EFFORT=$3
while IFS=$'\t' read -r id q; do
  [ -z "${id:-}" ] && continue
  out="out/live_ask/$TAG/$id.txt"
  echo "=== $TAG $id: $q"
  {
    echo "### model=$MODEL effort=$EFFORT"
    echo "### question: $q"
    echo "### tile: $TILE"
    echo
  } > "$out"
  start=$(date +%s)
  segmap ask -i "$TILE" --model "$MODEL" --effort "$EFFORT" "$q" >> "$out" 2>&1
  rc=$?
  echo "### exit=$rc elapsed=$(( $(date +%s) - start ))s" >> "$out"
done < out/live_ask/questions.txt
