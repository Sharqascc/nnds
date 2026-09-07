"""Package initialization for src."""

import os

from src.utils.seed import set_seed

# Use environment variable GLOBAL_SEED if set, otherwise default to 42
_global_seed = int(os.getenv("GLOBAL_SEED", "42"))
set_seed(_global_seed)

# Agentic test change
