"""Configure pytest for the project.

This file ensures that imports and paths are properly set up for testing.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def _isolate_kdp_features_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Isolate KDP feature-stats I/O so tests do not share CWD ``features_stats.json``.

    KDP's ``PreprocessingModel._init_stats`` constructs ``DatasetStatistics`` without
    forwarding ``features_stats_path`` / ``overwrite_stats``. That makes any leftover
    CWD ``features_stats.json`` (often from another test with different feature names)
    get loaded, which then causes ``KeyError: 'mean'`` when looking up the current
    features. Force a per-test stats path and overwrite so each test recomputes stats.
    """
    stats_path = tmp_path / "features_stats.json"

    # Drop any leftover CWD stats file from prior local runs / other suites.
    cwd_stats = Path.cwd() / "features_stats.json"
    if cwd_stats.is_file():
        cwd_stats.unlink()

    try:
        from kdp.stats import DatasetStatistics
    except ImportError:
        yield
        return

    original_init = DatasetStatistics.__init__

    def _patched_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        kwargs["features_stats_path"] = stats_path
        kwargs["overwrite_stats"] = True
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(DatasetStatistics, "__init__", _patched_init)

    # Also force PreprocessingModel to recompute even if something else set features_stats.
    try:
        from kdp.processor import PreprocessingModel
    except ImportError:
        yield
        return

    original_pm_init = PreprocessingModel.__init__

    def _patched_pm_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        kwargs.setdefault("overwrite_stats", True)
        kwargs.setdefault("features_stats_path", str(stats_path))
        original_pm_init(self, *args, **kwargs)

    monkeypatch.setattr(PreprocessingModel, "__init__", _patched_pm_init)
    yield

    if cwd_stats.is_file():
        cwd_stats.unlink()
