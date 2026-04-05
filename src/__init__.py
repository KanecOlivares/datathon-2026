"""Source package for the datathon project.

Provides compatibility aliases so numbered stage files can still be imported
through readable names such as ``src.config`` and ``src.etl``.
"""

from __future__ import annotations

import importlib
import sys


_MODULE_ALIASES = {
    "config": "01_config",
    "utils": "02_utils",
    "ingestion": "03_ingestion",
    "validation": "04_validation",
    "preprocessing": "05_preprocessing",
    "feature_engineering": "06_feature_engineering",
}


for alias, target in _MODULE_ALIASES.items():
    sys.modules[f"{__name__}.{alias}"] = importlib.import_module(f".{target}", __name__)
