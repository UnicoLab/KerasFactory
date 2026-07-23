"""Tests for FeatureSpaceAnomalyDetectionModel."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from kerasfactory.models.FeatureSpaceAnomalyDetectionModel import (
    FeatureSpaceAnomalyDetectionModel,
)


@pytest.fixture
def model() -> FeatureSpaceAnomalyDetectionModel:
    """Basic feature-space anomaly model."""
    return FeatureSpaceAnomalyDetectionModel(
        feature_space={"temperature": "numerical", "color": "cat_string"},
        numerical_threshold=2.0,
        business_rules={
            "temperature": [(">", 0), ("<", 100)],
            "color": [("in", ["red", "green", "blue"])],
        },
        numerical_method="z-score",
    )


def test_initialization(model: FeatureSpaceAnomalyDetectionModel) -> None:
    """Model stores configuration."""
    assert "temperature" in model.feature_space
    assert model.numerical_threshold == 2.0
    assert "temperature" in model.stat_layers


def test_fit_and_predict(model: FeatureSpaceAnomalyDetectionModel) -> None:
    """Fit initializes stats; predict returns global outputs."""
    train = tf.data.Dataset.from_tensor_slices(
        {
            "temperature": np.array([[20.0], [22.0], [21.0], [19.0]], dtype=np.float32),
            "color": np.array([["red"], ["green"], ["blue"], ["red"]]),
        },
    ).batch(2)
    model.fit(train)

    test = {
        "temperature": ops.convert_to_tensor([[21.0], [200.0]], dtype="float32"),
        "color": ops.convert_to_tensor([["red"], ["purple"]], dtype="string"),
    }
    preds = model(test)
    assert "global" in preds
    assert "temperature" in preds
    assert "color" in preds
    global_anom = ops.convert_to_numpy(preds["global"]["is_anomaly"])
    assert global_anom[1][0]


def test_get_config(model: FeatureSpaceAnomalyDetectionModel) -> None:
    """Config includes feature_space."""
    config = model.get_config()
    assert config["feature_space"]["temperature"] == "numerical"
    assert config["numerical_method"] == "z-score"
