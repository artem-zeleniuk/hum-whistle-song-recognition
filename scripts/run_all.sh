#!/bin/bash
# The whole study, from the download of the audio to the tables and figures.
#     bash scripts/run_all.sh [WORKERS]
# WORKERS is the number of CPU cores to use (default: all that the process may use). A GPU is used when
# PyTorch finds one, and the network trainings spread over several GPUs if there are several. The
# published run was split over several rented machines (results/stage_times.csv gives the
# time of every stage). On one machine with 64 cores and one 24 GB GPU expect about half a day, most of it
# for the networks and the fine-tuning. Every script can also be run on its own, see its --help.
set -eu
cd "$(dirname "$0")/.."
WORKERS=${1:-$(python -c "import os; print(len(os.sched_getaffinity(0)))")}
DATA=${DATA_DIR:-data/audio}
THREAD_VARIABLES="OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS NUMBA_NUM_THREADS"
one_thread() { for name in $THREAD_VARIABLES; do export "$name=1"; done; }
many_threads() { for name in $THREAD_VARIABLES; do unset "$name"; done; }

# 1. audio
python scripts/download_data.py --data-dir "$DATA"

# 2. features (one thread per worker process, otherwise the numeric libraries oversubscribe the cores)
one_thread
python scripts/extract_features.py --data-dir "$DATA" --workers "$WORKERS"
python scripts/extract_pitch.py --data-dir "$DATA" --workers "$WORKERS"
python scripts/extract_pitch.py --data-dir "$DATA" --workers "$WORKERS" --wide
python scripts/check_duplicates.py
python scripts/extract_ast.py --data-dir "$DATA" --workers "$WORKERS"
python scripts/extract_embeddings.py --data-dir "$DATA" --workers "$WORKERS"
python scripts/augmentation_diagnostic.py --data-dir "$DATA" --workers "$WORKERS"
many_threads

# 3. models
python scripts/run_classical.py --workers "$WORKERS"
python scripts/run_classical.py --checks --workers "$WORKERS"
python scripts/run_neural.py
python scripts/run_neural.py --checks
python scripts/run_variants.py
one_thread
python scripts/run_melody.py --data-dir "$DATA" --workers "$WORKERS"
many_threads

# 4. fine-tuning of MERT 95M: three learning rates with the first seed, then all seeds at the rate with the
#    best validation accuracy (runs that exist already are not repeated)
RATES="1e-5 3e-5 1e-4"
for rate in $RATES; do python scripts/finetune.py --encoder mert --lr "$rate" --seeds 0 --no-join; done
BEST=$(python - $RATES << 'PY'
import glob
import sys

import numpy as np

score = {}
for rate in sys.argv[1:]:
    files = glob.glob(f"results/finetune_parts/finetune__mert__none__lr{float(rate):g}__fold*_seed0.npz")
    score[rate] = float(np.mean([np.load(f)["val_accuracy"] for f in files]))
print(max(score, key=score.get))
PY
)
echo "learning rate chosen on validation: $BEST"
python scripts/finetune.py --encoder mert --lr "$BEST"

# 5. tables and figures
python scripts/make_report.py
