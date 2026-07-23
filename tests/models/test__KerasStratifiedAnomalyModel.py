"""Tests for KerasStratifiedAnomalyModel."""

from __future__ import annotations

import numpy as np
import pytest
from keras import ops

from kerasfactory.models.KerasStratifiedAnomalyModel import KerasStratifiedAnomalyModel


@pytest.fixture
def model() -> KerasStratifiedAnomalyModel:
    """Stratified keras model fixture."""
    return KerasStratifiedAnomalyModel(
        feature_space={"store_id": "string", "sales": "float"},
        key_features=["store_id"],
        target_features=["sales"],
        min_samples=2,
        threshold=2.0,
    )


def test_initialization(model: KerasStratifiedAnomalyModel) -> None:
    """Group configs generated from key features."""
    assert model.key_features == ["store_id"]
    assert "store_id" in next(iter(model.group_configs.values()))["keys"]


def test_call_shapes(model: KerasStratifiedAnomalyModel) -> None:
    """Call returns expected output keys and shapes."""
    inputs = {
        "store_id": ops.convert_to_tensor([["A"], ["A"], ["A"], ["A"]], dtype="string"),
        "sales": ops.convert_to_tensor(
            [[10.0], [11.0], [12.0], [100.0]], dtype="float32"
        ),
    }
    _ = model(inputs, training=True)
    outputs = model(inputs, training=False)
    assert "sales_anomaly_score" in outputs
    assert outputs["sales_anomaly_score"].shape == (4, 1)
    assert outputs["sales_is_anomaly"].shape == (4, 1)


def test_get_config(model: KerasStratifiedAnomalyModel) -> None:
    """Serialization config."""
    config = model.get_config()
    assert config["threshold"] == 2.0
    restored = KerasStratifiedAnomalyModel.from_config(config)
    assert restored.target_features == ["sales"]
