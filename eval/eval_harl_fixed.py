"""Algorithm-neutral fixed evaluation entry; legacy MAPPO name remains compatible."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_harl_mappo_fixed import main


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
