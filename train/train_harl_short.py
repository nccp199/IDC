"""Unified formal short-training CLI for MAPPO/HAPPO and explicit critic selection.

The legacy ``train_harl_mappo_short`` module remains import-compatible while
this algorithm-neutral filename is the preferred command-line entry.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_harl_mappo_short import main


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
