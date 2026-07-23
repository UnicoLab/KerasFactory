"""Tests for StratifiedAnomalyDetectionModel."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from kerasfactory.models.StratifiedAnomalyDetectionModel import (
    StratifiedAnomalyDetectionModel,
)


def test_invalid_config() -> None:
    """Missing stratified keys raise."""
    with pytest.raises(ValueError):
        StratifiedAnomalyDetectionModel(
            feature_space={"a": "numerical", "b": "cat_string"},
            stratified_config={"keys": ["b"]},
        )


def test_fit_predict_groups() -> None:
    """Fit builds per-group models and predict returns group results."""
    model = StratifiedAnomalyDetectionModel(
        feature_space={"category": "cat_string", "value": "numerical"},
        stratified_config={
            "keys": ["category"],
            "targets": ["value"],
            "min_samples_per_group": 2,
            "fallback_strategy": "global",
        },
        numerical_threshold=2.0,
    )
    train = tf.data.Dataset.from_tensor_slices(
        {
            "category": np.array([["A"], ["A"], ["B"], ["B"], ["A"], ["B"]]),
            "value": np.array(
                [[1.0], [1.1], [5.0], [5.2], [0.9], [4.8]], dtype=np.float32
            ),
        },
    ).batch(2)
    model.fit(train)
    assert len(model.group_models) >= 1

    test = tf.data.Dataset.from_tensor_slices(
        {
            "category": np.array([["A"], ["B"]]),
            "value": np.array([[1.0], [50.0]], dtype=np.float32),
        },
    ).batch(2)
    preds = model.predict(test)
    assert isinstance(preds, dict)
    assert len(preds) >= 1
