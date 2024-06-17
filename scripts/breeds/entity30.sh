#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Provide output dir as an argument!"
  exit 1
fi

echo ">>> Saving outputs to $1"

if [ -d "$1" ]; then
  mkdir -p "$1"
fi

if [ -f "$1/entity30_neighbors.pth" ]; then
  echo "Nearest neighbors already calculated, skipping the retrieval"
else
  python find_neighbors.py --cfg_file configs/breeds/coarse2fine/entity30.yaml --override_cfg OUTPUT_DIR "$1" --output_dir "$1" --use_faiss
fi

python -W ignore main.py --port 8080 --cfg_file configs/breeds/coarse2fine/entity30.yaml --override_cfg OUTPUT_DIR "$1/run1" NEIGHBORS "$1/entity30_neighbors.pth"
