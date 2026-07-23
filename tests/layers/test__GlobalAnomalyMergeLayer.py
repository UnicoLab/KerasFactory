"""Tests for GlobalAnomalyMergeLayer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np
import pytest
from keras import ops

from kerasfactory.layers.GlobalAnomalyMergeLayer import GlobalAnomalyMergeLayer


@pytest.fixture
def mock_outputs() -> dict[str, dict[str, Any]]:
    """Mock per-feature anomaly outputs."""
    return {
        "temperature": {
            "proba": ops.convert_to_tensor([[30.0], [80.0]], dtype="float32"),
            "score": ops.convert_to_tensor([[1.5], [2.5]], dtype="float32"),
            "anomaly": ops.convert_to_tensor([[False], [True]], dtype="bool"),
            "reason": ops.convert_to_tensor(
                ["Normal temperature", "Temperature too high"],
                dtype="string",
            ),
        },
        "color": {
            "proba": ops.convert_to_tensor([[0.0], [100.0]], dtype="float32"),
            "score": ops.convert_to_tensor([[0.5], [3.0]], dtype="float32"),
            "anomaly": ops.convert_to_tensor([[False], [True]], dtype="bool"),
            "reason": ops.convert_to_tensor(
                ["Valid color", "Invalid color"],
                dtype="string",
            ),
        },
    }


def test_invalid_merge_strategy() -> None:
    """Invalid strategy raises."""
    with pytest.raises(ValueError):
        GlobalAnomalyMergeLayer(merge_strategy="invalid")


def test_max_strategy(mock_outputs: dict[str, dict[str, Any]]) -> None:
    """Max merge combines scores correctly."""
    layer = GlobalAnomalyMergeLayer(merge_strategy="max")
    outputs = layer(mock_outputs)
    assert set(outputs) >= {"probability", "score", "is_anomaly", "reason", "details"}
    probability = ops.convert_to_numpy(outputs["probability"])
    score = ops.convert_to_numpy(outputs["score"])
    is_anomaly = ops.convert_to_numpy(outputs["is_anomaly"])
    assert np.allclose(probability, [[30.0], [100.0]])
    assert np.allclose(score, [[1.5], [3.0]])
    assert not is_anomaly[0][0]
    assert is_anomaly[1][0]


def test_mean_strategy(mock_outputs: dict[str, dict[str, Any]]) -> None:
    """Mean merge averages scores."""
    layer = GlobalAnomalyMergeLayer(merge_strategy="mean")
    outputs = layer(mock_outputs)
    assert np.allclose(ops.convert_to_numpy(outputs["probability"]), [[15.0], [90.0]])
    assert np.allclose(ops.convert_to_numpy(outputs["score"]), [[1.0], [2.75]])


def test_reason_and_shapes(mock_outputs: dict[str, dict[str, Any]]) -> None:
    """Reasons and shapes are consistent."""
    layer = GlobalAnomalyMergeLayer(merge_strategy="max")
    outputs = layer(mock_outputs)
    reasons = ops.convert_to_numpy(outputs["reason"])
    assert reasons[0].decode() == "No anomalies detected"
    assert "color" in reasons[1].decode()
    assert outputs["probability"].shape == (2, 1)
    assert outputs["reason"].shape == (2,)


def test_serialization(mock_outputs: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    """Config and functional save/load."""
    layer = GlobalAnomalyMergeLayer(merge_strategy="max")
    config = layer.get_config()
    assert config["merge_strategy"] == "max"
    new_layer = GlobalAnomalyMergeLayer.from_config(config)
    orig = layer(mock_outputs)
    new = new_layer(mock_outputs)
    np.testing.assert_allclose(
        ops.convert_to_numpy(orig["score"]),
        ops.convert_to_numpy(new["score"]),
        rtol=1e-5,
    )

    # Functional model with numeric stand-ins for reasons
    inputs = {
        "temperature": keras.Input(
            shape=(1,), batch_size=2, dtype="float32", name="temperature"
        ),
        "color": keras.Input(shape=(1,), batch_size=2, dtype="float32", name="color"),
    }
    outputs = layer(
        {
            "temperature": {
                "proba": inputs["temperature"],
                "score": inputs["temperature"],
                "anomaly": ops.cast(ops.greater(inputs["temperature"], 0.5), "bool"),
                "reason": ops.zeros_like(inputs["temperature"], dtype="int32"),
            },
            "color": {
                "proba": inputs["color"],
                "score": inputs["color"],
                "anomaly": ops.cast(ops.greater(inputs["color"], 0.5), "bool"),
                "reason": ops.zeros_like(inputs["color"], dtype="int32"),
            },
        },
    )
    model = keras.Model(inputs=inputs, outputs=outputs)
    path = str(tmp_path / "global_merge.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    test_input = {
        "temperature": ops.convert_to_tensor([[0.3], [0.7]], dtype="float32"),
        "color": ops.convert_to_tensor([[0.4], [0.8]], dtype="float32"),
    }
    np.testing.assert_allclose(
        ops.convert_to_numpy(model(test_input)["score"]),
        ops.convert_to_numpy(loaded(test_input)["score"]),
        rtol=1e-5,
    )
