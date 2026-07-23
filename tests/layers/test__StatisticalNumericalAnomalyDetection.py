"""Tests for StatisticalNumericalAnomalyDetection."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from kerasfactory.layers.StatisticalNumericalAnomalyDetection import (
    StatisticalNumericalAnomalyDetection,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Temporary directory fixture."""
    return tmp_path


def test_initialization_defaults() -> None:
    """Test default initialization."""
    layer = StatisticalNumericalAnomalyDetection()
    assert layer.method == "z-score"
    assert layer.threshold == 2.0


def test_invalid_method() -> None:
    """Invalid method raises ValueError."""
    with pytest.raises(ValueError):
        StatisticalNumericalAnomalyDetection(method="invalid")  # type: ignore[arg-type]


def test_zscore_call() -> None:
    """Z-score detects outliers beyond threshold."""
    layer = StatisticalNumericalAnomalyDetection(method="z-score", threshold=2.0)
    layer.initialize_from_stats(mean=50.0, std=10.0)

    normal = layer(ops.convert_to_tensor([[55.0]], dtype="float32"))
    assert float(ops.convert_to_numpy(normal["score"])[0][0]) < 2.0
    assert not bool(ops.convert_to_numpy(normal["anomaly"])[0][0])

    anomalous = layer(ops.convert_to_tensor([[80.0]], dtype="float32"))
    assert float(ops.convert_to_numpy(anomalous["score"])[0][0]) > 2.0
    assert bool(ops.convert_to_numpy(anomalous["anomaly"])[0][0])


def test_iqr_and_mad() -> None:
    """IQR and MAD methods produce expected keys."""
    iqr = StatisticalNumericalAnomalyDetection(method="iqr", threshold=1.5)
    iqr.initialize_from_stats({"q1": 40.0, "q3": 60.0})
    out = iqr(ops.convert_to_tensor([[100.0]], dtype="float32"))
    assert "score" in out and "anomaly" in out

    mad = StatisticalNumericalAnomalyDetection(method="mad", threshold=2.0)
    mad.initialize_from_stats({"median": 50.0, "mad": 5.0})
    out = mad(ops.convert_to_tensor([[70.0]], dtype="float32"))
    assert bool(ops.convert_to_numpy(out["anomaly"])[0][0])


def test_batch_shapes() -> None:
    """Batch processing preserves shapes."""
    layer = StatisticalNumericalAnomalyDetection(threshold=2.0)
    layer.initialize_from_stats({"mean": 50.0, "std": 10.0})
    out = layer(ops.convert_to_tensor([[45.0], [50.0], [80.0]], dtype="float32"))
    assert out["score"].shape == (3, 1)
    assert out["proba"].shape == (3, 1)
    assert out["anomaly"].shape == (3, 1)


def test_serialization_and_keras_roundtrip(temp_dir: Path) -> None:
    """Config round-trip and .keras save/load."""
    layer = StatisticalNumericalAnomalyDetection(method="z-score", threshold=2.0)
    layer.initialize_from_stats(mean=50.0, std=10.0)
    config = layer.get_config()
    assert config["threshold"] == 2.0

    new_layer = StatisticalNumericalAnomalyDetection.from_config(config)
    new_layer.initialize_from_stats(mean=50.0, std=10.0)

    x = ops.convert_to_tensor([[55.0], [80.0]], dtype="float32")
    np.testing.assert_allclose(
        ops.convert_to_numpy(layer(x)["score"]),
        ops.convert_to_numpy(new_layer(x)["score"]),
        rtol=1e-5,
    )

    inputs = keras.Input(shape=(1,), dtype="float32")
    model = keras.Model(inputs=inputs, outputs=layer(inputs))
    path = str(temp_dir / "stat_num.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    orig = model(x)
    loaded_out = loaded(x)
    np.testing.assert_allclose(
        ops.convert_to_numpy(orig["score"]),
        ops.convert_to_numpy(loaded_out["score"]),
        rtol=1e-5,
    )
