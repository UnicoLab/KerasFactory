"""Tests for MultiFeatureBusinessRulesLayer."""

from __future__ import annotations

import keras
import numpy as np
import pytest
from keras import ops

from kerasfactory.layers.MultiFeatureBusinessRulesLayer import (
    MultiFeatureBusinessRulesLayer,
)


@pytest.fixture
def rules() -> dict:
    """Sample temperature/season rules."""
    return {
        "temperature_season": {
            "features": ["temperature", "season"],
            "conditions": [
                {
                    "when": {"season": "winter"},
                    "require": {"temperature": [("<", 25.0)]},
                },
                {
                    "when": {"season": "summer"},
                    "require": {"temperature": [(">", 15.0)]},
                },
            ],
        },
    }


@pytest.fixture
def multi_rules() -> dict:
    """Rules covering winter temp and summer humidity."""
    return {
        "winter_temp_rule": {
            "features": ["temperature", "season"],
            "conditions": [
                {
                    "when": {"season": "winter"},
                    "require": {"temperature": [("<", 25.0)]},
                },
            ],
        },
        "summer_humidity_rule": {
            "features": ["temperature", "humidity", "season"],
            "conditions": [
                {
                    "when": {"season": "summer", "temperature": (">=", 30.0)},
                    "require": {"humidity": [("<", 70.0)]},
                },
            ],
        },
    }


def test_invalid_rules() -> None:
    """Missing features key raises."""
    with pytest.raises(ValueError):
        MultiFeatureBusinessRulesLayer(
            rules={"bad": {"conditions": []}},
            feature_types={"a": "numerical"},
        )


def test_invalid_feature_type(rules: dict) -> None:
    """Unknown feature_type raises."""
    with pytest.raises(ValueError, match="Invalid feature_type"):
        MultiFeatureBusinessRulesLayer(
            rules=rules,
            feature_types={"temperature": "invalid", "season": "categorical"},
        )


def test_rule_violation(rules: dict) -> None:
    """Winter with high temperature is anomalous."""
    layer = MultiFeatureBusinessRulesLayer(
        rules=rules,
        feature_types={"temperature": "numerical", "season": "categorical"},
    )
    outputs = layer(
        {
            "temperature": ops.convert_to_tensor([[5.0], [30.0]], dtype="float32"),
            "season": ops.convert_to_tensor([["winter"], ["winter"]], dtype="string"),
        },
    )
    anomalies = ops.convert_to_numpy(outputs["business_anomaly"])
    assert not anomalies[0][0]
    assert anomalies[1][0]
    assert "business_score" in outputs
    assert "business_reason" in outputs


def test_summer_humidity_and_non_matching(multi_rules: dict) -> None:
    """Summer humidity rule and non-matching spring."""
    layer = MultiFeatureBusinessRulesLayer(
        rules=multi_rules,
        feature_types={
            "temperature": "numerical",
            "season": "categorical",
            "humidity": "numerical",
        },
    )
    summer = layer(
        {
            "temperature": ops.convert_to_tensor([[35.0], [35.0]], dtype="float32"),
            "season": ops.convert_to_tensor([["summer"], ["summer"]], dtype="string"),
            "humidity": ops.convert_to_tensor([[80.0], [60.0]], dtype="float32"),
        },
    )
    assert bool(ops.convert_to_numpy(summer["business_anomaly"])[0][0])
    assert not bool(ops.convert_to_numpy(summer["business_anomaly"])[1][0])
    assert tuple(ops.convert_to_numpy(summer["rule_violations"]).shape) == (2, 2)

    spring = layer(
        {
            "temperature": ops.convert_to_tensor([[30.0]], dtype="float32"),
            "season": ops.convert_to_tensor([["spring"]], dtype="string"),
            "humidity": ops.convert_to_tensor([[80.0]], dtype="float32"),
        },
    )
    assert not bool(ops.convert_to_numpy(spring["business_anomaly"])[0][0])


def test_serialization(rules: dict) -> None:
    """from_config recreates the layer."""
    layer = MultiFeatureBusinessRulesLayer(
        rules=rules,
        feature_types={"temperature": "numerical", "season": "categorical"},
    )
    config = layer.get_config()
    restored = MultiFeatureBusinessRulesLayer.from_config(config)
    assert restored.rules == rules
    inputs = {
        "temperature": ops.convert_to_tensor([[30.0]], dtype="float32"),
        "season": ops.convert_to_tensor([["winter"]], dtype="string"),
    }
    np.testing.assert_array_equal(
        ops.convert_to_numpy(layer(inputs)["business_anomaly"]),
        ops.convert_to_numpy(restored(inputs)["business_anomaly"]),
    )


def test_keras_serialize_handles_list_rules(multi_rules: dict) -> None:
    """Serialize/deserialize must tolerate tuples becoming lists."""
    layer = MultiFeatureBusinessRulesLayer(
        rules=multi_rules,
        feature_types={
            "temperature": "numerical",
            "season": "categorical",
            "humidity": "numerical",
        },
    )
    serialized = keras.saving.serialize_keras_object(layer)
    restored = keras.saving.deserialize_keras_object(
        serialized,
        custom_objects={
            "MultiFeatureBusinessRulesLayer": MultiFeatureBusinessRulesLayer,
        },
    )
    inputs = {
        "temperature": ops.convert_to_tensor([[35.0]], dtype="float32"),
        "season": ops.convert_to_tensor([["summer"]], dtype="string"),
        "humidity": ops.convert_to_tensor([[80.0]], dtype="float32"),
    }
    np.testing.assert_array_equal(
        ops.convert_to_numpy(layer(inputs)["business_anomaly"]),
        ops.convert_to_numpy(restored(inputs)["business_anomaly"]),
    )
