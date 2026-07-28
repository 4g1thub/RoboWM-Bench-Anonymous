#!/usr/bin/env bash

# Pass a different dataset root as the first argument when using another
# local directory layout.
DATASET_ROOT="${1:-./eval_videos/025}"

python tools/convertv21.py \
	--repo-id="001" \
	--root="${DATASET_ROOT}"
