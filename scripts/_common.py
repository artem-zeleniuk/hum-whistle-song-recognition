"""Small helpers shared by the command-line scripts."""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_START = time.time()


def log(message):
    """Print one line with the minutes since the start of the script."""
    print(f"[{(time.time() - _START) / 60:6.1f} min] {message}", flush=True)


def default_workers():
    """CPUs this process may use (the affinity mask, which is what a container is given)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def add_common_arguments(parser):
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, "cache"),
                        help="folder for extracted features (large, not part of the repository)")
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "results"),
                        help="folder for predictions, tables and logs")
    parser.add_argument("--workers", type=int, default=default_workers(), help="number of worker processes")
