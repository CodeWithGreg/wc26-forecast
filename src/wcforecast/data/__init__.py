"""Data access layer: historical results and World Cup 2026 structure."""

from wcforecast.data.results import load_results  # noqa: F401
from wcforecast.data.wc2026 import (  # noqa: F401
    load_bracket,
    load_group_fixtures,
    load_groups,
)
