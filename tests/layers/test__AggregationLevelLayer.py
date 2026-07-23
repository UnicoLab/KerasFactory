"""Tests for AggregationLevelLayer."""

from __future__ import annotations

import keras
import numpy as np
import pytest
from keras import ops

from kerasfactory.layers.AggregationLevelLayer import AggregationLevelLayer


@pytest.fixture
def layer() -> AggregationLevelLayer:
    """Aggregation layer with lengths/weights/appearance levels."""
    feature_types = {
        "length_a": "numerical",
        "length_b": "numerical",
        "weight_a": "numerical",
        "weight_b": "numerical",
        "color": "categorical",
        "material": "categorical",
    }
    aggregation_levels = {
        "lengths": {
            "features": ["length_a", "length_b"],
            "stats": {
                "method": "z-score",
                "threshold": 2.0,
                "aggregation_function": "mean",
            },
        },
        "weights": {
            "features": ["weight_a", "weight_b"],
            "stats": {
                "method": "z-score",
                "threshold": 2.5,
                "aggregation_function": "max",
            },
            "rules": {
                "weight_ratio_rule": {
                    "conditions": [
                        {
                            "when": {"weight_a": (">", 10.0)},
                            "require": {"weight_b": [(">", 5.0)]},
                        },
                    ],
                },
            },
        },
        "appearance": {
            "features": ["color", "material"],
            "rules": {
                "valid_combinations": {
                    "conditions": [
                        {
                            "when": {"color": "red"},
                            "require": {"material": [("!=", "plastic")]},
                        },
                    ],
                },
            },
        },
    }
    return AggregationLevelLayer(
        aggregation_levels=aggregation_levels,
        feature_types=feature_types,
        name="test_aggregation_layer",
    )


def test_initialization(layer: AggregationLevelLayer) -> None:
    """Rule layers created for levels that define rules."""
    assert "weights" in layer.rule_layers
    assert "appearance" in layer.rule_layers
    assert "lengths" not in layer.rule_layers
    assert layer.feature_to_level["length_a"] == "lengths"


def test_duplicate_feature_raises() -> None:
    """Feature in two levels raises."""
    with pytest.raises(ValueError):
        AggregationLevelLayer(
            aggregation_levels={
                "a": {"features": ["x"]},
                "b": {"features": ["x"]},
            },
            feature_types={"x": "numerical"},
        )


def test_invalid_aggregation_function() -> None:
    """Unsupported aggregation_function raises at init."""
    with pytest.raises(ValueError, match="aggregation_function"):
        AggregationLevelLayer(
            aggregation_levels={
                "a": {
                    "features": ["x"],
                    "stats": {
                        "threshold": 1.0,
                        "aggregation_function": "median",
                    },
                },
            },
            feature_types={"x": "numerical"},
        )


def test_call_with_stats(layer: AggregationLevelLayer) -> None:
    """Stats aggregation and rule evaluation."""
    inputs = {
        "length_a": ops.convert_to_tensor([[1.0], [5.0]], dtype="float32"),
        "length_b": ops.convert_to_tensor([[2.0], [6.0]], dtype="float32"),
        "weight_a": ops.convert_to_tensor([[11.0], [8.0]], dtype="float32"),
        "weight_b": ops.convert_to_tensor([[4.0], [7.0]], dtype="float32"),
        "color": ops.convert_to_tensor([["red"], ["blue"]], dtype="string"),
        "material": ops.convert_to_tensor([["metal"], ["plastic"]], dtype="string"),
    }
    feature_stats = {
        "length_a": ops.convert_to_tensor([[0.5], [2.5]], dtype="float32"),
        "length_b": ops.convert_to_tensor([[0.3], [2.0]], dtype="float32"),
        "weight_a": ops.convert_to_tensor([[1.0], [0.5]], dtype="float32"),
        "weight_b": ops.convert_to_tensor([[0.8], [3.0]], dtype="float32"),
        "color": ops.convert_to_tensor([[0.0], [0.0]], dtype="float32"),
        "material": ops.convert_to_tensor([[0.0], [0.0]], dtype="float32"),
    }
    # Keyword arg preferred so Keras __call__ does not confuse with training
    outputs = layer(inputs, feature_stats=feature_stats)

    assert abs(float(ops.convert_to_numpy(outputs["lengths_score"])[0][0]) - 0.4) < 1e-5
    assert (
        abs(float(ops.convert_to_numpy(outputs["lengths_score"])[1][0]) - 2.25) < 1e-5
    )
    assert not bool(ops.convert_to_numpy(outputs["lengths_anomaly"])[0][0])
    assert bool(ops.convert_to_numpy(outputs["lengths_anomaly"])[1][0])

    assert abs(float(ops.convert_to_numpy(outputs["weights_score"])[0][0]) - 1.0) < 1e-5
    assert abs(float(ops.convert_to_numpy(outputs["weights_score"])[1][0]) - 3.0) < 1e-5
    assert not bool(ops.convert_to_numpy(outputs["weights_anomaly"])[0][0])
    assert bool(ops.convert_to_numpy(outputs["weights_anomaly"])[1][0])
    assert bool(ops.convert_to_numpy(outputs["weights_rule_anomaly"])[0][0])

    assert "aggregation_anomaly" in outputs


def test_call_without_feature_stats(layer: AggregationLevelLayer) -> None:
    """Rules-only path works with empty / missing feature_stats."""
    inputs = {
        "length_a": ops.convert_to_tensor([[1.0]], dtype="float32"),
        "length_b": ops.convert_to_tensor([[1.0]], dtype="float32"),
        "weight_a": ops.convert_to_tensor([[11.0]], dtype="float32"),
        "weight_b": ops.convert_to_tensor([[4.0]], dtype="float32"),
        "color": ops.convert_to_tensor([["blue"]], dtype="string"),
        "material": ops.convert_to_tensor([["metal"]], dtype="string"),
    }
    outputs = layer(inputs)
    assert bool(ops.convert_to_numpy(outputs["weights_rule_anomaly"])[0][0])
    assert not bool(ops.convert_to_numpy(outputs["lengths_anomaly"])[0][0])
    assert bool(ops.convert_to_numpy(outputs["aggregation_anomaly"])[0][0])


def test_appearance_rule_anomaly(layer: AggregationLevelLayer) -> None:
    """Red plastic violates appearance rule and global aggregation."""
    inputs = {
        "length_a": ops.convert_to_tensor([[1.0]], dtype="float32"),
        "length_b": ops.convert_to_tensor([[1.0]], dtype="float32"),
        "weight_a": ops.convert_to_tensor([[1.0]], dtype="float32"),
        "weight_b": ops.convert_to_tensor([[1.0]], dtype="float32"),
        "color": ops.convert_to_tensor([["red"]], dtype="string"),
        "material": ops.convert_to_tensor([["plastic"]], dtype="string"),
    }
    outputs = layer(inputs, feature_stats={})
    assert bool(ops.convert_to_numpy(outputs["appearance_rule_anomaly"])[0][0])
    assert bool(ops.convert_to_numpy(outputs["aggregation_anomaly"])[0][0])


def test_serialization(layer: AggregationLevelLayer) -> None:
    """Config round-trip."""
    config = layer.get_config()
    restored = AggregationLevelLayer.from_config(config)
    assert restored.aggregation_levels == layer.aggregation_levels


def test_keras_serialize_deserialize(layer: AggregationLevelLayer) -> None:
    """Keras object serialization round-trip."""
    serialized = keras.saving.serialize_keras_object(layer)
    restored = keras.saving.deserialize_keras_object(
        serialized,
        custom_objects={"AggregationLevelLayer": AggregationLevelLayer},
    )
    assert restored.aggregation_levels == layer.aggregation_levels
    inputs = {
        "length_a": ops.convert_to_tensor([[1.0]], dtype="float32"),
        "length_b": ops.convert_to_tensor([[1.0]], dtype="float32"),
        "weight_a": ops.convert_to_tensor([[11.0]], dtype="float32"),
        "weight_b": ops.convert_to_tensor([[4.0]], dtype="float32"),
        "color": ops.convert_to_tensor([["blue"]], dtype="string"),
        "material": ops.convert_to_tensor([["metal"]], dtype="string"),
    }
    np.testing.assert_array_equal(
        ops.convert_to_numpy(layer(inputs)["aggregation_anomaly"]),
        ops.convert_to_numpy(restored(inputs)["aggregation_anomaly"]),
    )
