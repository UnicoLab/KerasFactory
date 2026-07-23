"""Tests for GroupStatisticsLayer."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pytest
from keras import ops

from kerasfactory.layers.GroupStatisticsLayer import GroupStatisticsLayer


@pytest.fixture
def layer() -> GroupStatisticsLayer:
    """Single-key group statistics layer fixture."""
    return GroupStatisticsLayer(
        group_configs={
            "store": {"keys": ["store_id"], "threshold": 2.0},
        },
        target_features=["sales"],
        min_samples=2,
    )


@pytest.fixture
def multi_key_layer() -> GroupStatisticsLayer:
    """Multi-feature key group statistics layer."""
    return GroupStatisticsLayer(
        group_configs={
            "product_store": {
                "keys": ["product_category", "store_location"],
                "threshold": 2.0,
            },
        },
        target_features=["sales", "inventory"],
        min_samples=3,
    )


def _as_numpy(tensor: object) -> np.ndarray:
    """Convert a Keras/TF tensor to numpy."""
    return np.asarray(ops.convert_to_numpy(tensor))


def test_invalid_params() -> None:
    """Invalid configs raise ValueError."""
    with pytest.raises(ValueError, match="group_configs"):
        GroupStatisticsLayer(group_configs={}, target_features=["sales"])
    with pytest.raises(ValueError, match="target_features"):
        GroupStatisticsLayer(
            group_configs={"g": {"keys": ["a"]}},
            target_features=[],
        )
    with pytest.raises(ValueError, match="min_samples"):
        GroupStatisticsLayer(
            group_configs={"g": {"keys": ["a"]}},
            target_features=["sales"],
            min_samples=0,
        )
    with pytest.raises(ValueError, match="fallback_strategy"):
        GroupStatisticsLayer(
            group_configs={"g": {"keys": ["a"]}},
            target_features=["sales"],
            fallback_strategy="nearest",
        )
    with pytest.raises(ValueError, match="keys"):
        GroupStatisticsLayer(
            group_configs={"g": {"keys": []}},
            target_features=["sales"],
        )


def test_missing_input_features(layer: GroupStatisticsLayer) -> None:
    """Missing target or key features raise."""
    with pytest.raises(ValueError, match="target_features"):
        layer({"store_id": ops.convert_to_tensor(["A"], dtype="string")})
    with pytest.raises(ValueError, match="missing keys"):
        layer({"sales": ops.convert_to_tensor([1.0], dtype="float32")})


def test_train_and_infer(layer: GroupStatisticsLayer) -> None:
    """Training updates stats; inference scores outliers."""
    inputs = {
        "store_id": ops.convert_to_tensor(
            [["A"], ["A"], ["A"], ["A"]],
            dtype="string",
        ),
        "sales": ops.convert_to_tensor(
            [[10.0], [12.0], [11.0], [100.0]],
            dtype="float32",
        ),
    }
    _ = layer(inputs, training=True)
    assert float(_as_numpy(layer.global_count)) >= 4.0

    outputs = layer(inputs, training=False)
    assert "sales" in outputs
    assert "anomaly_score" in outputs["sales"]
    scores = _as_numpy(outputs["sales"]["anomaly_score"])
    assert scores.shape == (4,)
    assert scores[3] > scores[0]


def test_multi_group_keys_independent_stats(
    multi_key_layer: GroupStatisticsLayer,
) -> None:
    """Distinct key combinations keep independent running statistics."""
    # Combination electronics|north ~ sales 10; furniture|south ~ sales 1000.
    train = {
        "product_category": ops.convert_to_tensor(
            [
                "electronics",
                "electronics",
                "electronics",
                "furniture",
                "furniture",
                "furniture",
            ],
            dtype="string",
        ),
        "store_location": ops.convert_to_tensor(
            ["north", "north", "north", "south", "south", "south"],
            dtype="string",
        ),
        "sales": ops.convert_to_tensor(
            [10.0, 11.0, 12.0, 1000.0, 1010.0, 990.0],
            dtype="float32",
        ),
        "inventory": ops.convert_to_tensor(
            [100.0, 105.0, 95.0, 500.0, 510.0, 490.0],
            dtype="float32",
        ),
    }
    _ = multi_key_layer(train, training=True)

    combo_ids = list(multi_key_layer.combination_stats.keys())
    assert len(combo_ids) == 2
    assert any("electronics|north" in c for c in combo_ids)
    assert any("furniture|south" in c for c in combo_ids)

    # electronics|north sales=10 is normal for that group; furniture-scale value
    # would be anomalous if stats were wrongly pooled.
    infer = {
        "product_category": ops.convert_to_tensor(
            ["electronics", "electronics"],
            dtype="string",
        ),
        "store_location": ops.convert_to_tensor(["north", "north"], dtype="string"),
        "sales": ops.convert_to_tensor([11.0, 1000.0], dtype="float32"),
        "inventory": ops.convert_to_tensor([100.0, 100.0], dtype="float32"),
    }
    out = multi_key_layer(infer, training=False)
    scores = _as_numpy(out["sales"]["anomaly_score"])
    assert scores[1] > scores[0]
    assert bool(_as_numpy(out["sales"]["is_anomaly"])[1])


def test_insufficient_samples_uses_global_fallback() -> None:
    """Under-sampled combinations fall back to global statistics."""
    layer = GroupStatisticsLayer(
        group_configs={"store": {"keys": ["store_id"], "threshold": 2.0}},
        target_features=["sales"],
        min_samples=5,
    )
    # Tight global distribution from store B only.
    train_b = {
        "store_id": ops.convert_to_tensor(
            ["B", "B", "B", "B", "B", "B"],
            dtype="string",
        ),
        "sales": ops.convert_to_tensor(
            [100.0, 101.0, 99.0, 102.0, 98.0, 100.0],
            dtype="float32",
        ),
    }
    _ = layer(train_b, training=True)

    # Unseen / under-sampled store A: no local stats → global fallback.
    infer = {
        "store_id": ops.convert_to_tensor(["A"], dtype="string"),
        "sales": ops.convert_to_tensor([11.0], dtype="float32"),
    }
    out = layer(infer, training=False)
    combo_a = next(
        (c for c in layer.combination_stats if c.endswith("::A")),
        None,
    )
    if combo_a is not None:
        assert float(layer.combination_stats[combo_a]["count"]) < 5.0

    reasons = _as_numpy(out["sales"]["reason"])
    reason_text = (
        reasons[0].decode() if isinstance(reasons[0], bytes) else str(reasons[0])
    )
    assert bool(_as_numpy(out["sales"]["is_anomaly"])[0])
    assert "global" in reason_text.lower()
    assert float(_as_numpy(out["sales"]["anomaly_score"])[0]) > 2.0


def test_training_vs_inference_does_not_update_on_infer(
    layer: GroupStatisticsLayer,
) -> None:
    """Inference must not mutate running statistics."""
    inputs = {
        "store_id": ops.convert_to_tensor(["A", "A", "A"], dtype="string"),
        "sales": ops.convert_to_tensor([10.0, 11.0, 12.0], dtype="float32"),
    }
    _ = layer(inputs, training=True)
    count_after_train = float(_as_numpy(layer.global_count))
    combo_id = next(iter(layer.combination_stats))
    combo_count = float(layer.combination_stats[combo_id]["count"])

    noisy = {
        "store_id": ops.convert_to_tensor(["A", "A", "A"], dtype="string"),
        "sales": ops.convert_to_tensor([999.0, 998.0, 997.0], dtype="float32"),
    }
    _ = layer(noisy, training=False)
    assert float(_as_numpy(layer.global_count)) == count_after_train
    assert float(layer.combination_stats[combo_id]["count"]) == combo_count


def test_empty_batch(layer: GroupStatisticsLayer) -> None:
    """Empty batch returns empty outputs without crashing."""
    inputs = {
        "store_id": ops.convert_to_tensor([], dtype="string"),
        "sales": ops.convert_to_tensor([], dtype="float32"),
    }
    # Build with a non-empty batch first so weights exist.
    _ = layer(
        {
            "store_id": ops.convert_to_tensor(["A"], dtype="string"),
            "sales": ops.convert_to_tensor([1.0], dtype="float32"),
        },
        training=True,
    )
    out = layer(inputs, training=False)
    assert _as_numpy(out["sales"]["anomaly_score"]).shape == (0,)
    assert _as_numpy(out["sales"]["is_anomaly"]).shape == (0,)


def test_single_sample(layer: GroupStatisticsLayer) -> None:
    """Single-sample batch trains and scores."""
    sample = {
        "store_id": ops.convert_to_tensor(["A"], dtype="string"),
        "sales": ops.convert_to_tensor([42.0], dtype="float32"),
    }
    out_train = layer(sample, training=True)
    assert _as_numpy(out_train["sales"]["anomaly_score"]).shape == (1,)
    out_infer = layer(sample, training=False)
    assert _as_numpy(out_infer["sales"]["anomaly_score"]).shape == (1,)


def test_multiple_group_configs_processed_mask() -> None:
    """First matching group config wins via processed_mask order."""
    layer = GroupStatisticsLayer(
        group_configs={
            "fine": {"keys": ["store_id", "region"], "threshold": 2.0},
            "coarse": {"keys": ["store_id"], "threshold": 2.0},
        },
        target_features=["sales"],
        min_samples=2,
    )
    train = {
        "store_id": ops.convert_to_tensor(["A", "A", "A", "A"], dtype="string"),
        "region": ops.convert_to_tensor(["N", "N", "N", "N"], dtype="string"),
        "sales": ops.convert_to_tensor([10.0, 11.0, 12.0, 50.0], dtype="float32"),
    }
    _ = layer(train, training=True)
    # Both schemas should have combination entries for the batch keys.
    assert any(c.startswith("fine::") for c in layer.combination_stats)
    assert any(c.startswith("coarse::") for c in layer.combination_stats)

    out = layer(train, training=False)
    reasons = _as_numpy(out["sales"]["reason"])
    # Anomalous sample reason should reference the first (fine) group when enough
    # samples exist.
    anomaly_flags = _as_numpy(out["sales"]["is_anomaly"])
    if anomaly_flags.any():
        idx = int(np.argmax(anomaly_flags))
        text = (
            reasons[idx].decode()
            if isinstance(reasons[idx], bytes)
            else str(reasons[idx])
        )
        assert "fine" in text or "global" in text.lower()


def test_serialization_get_config_from_config(layer: GroupStatisticsLayer) -> None:
    """Config round-trip preserves parameters and combination stats."""
    train = {
        "store_id": ops.convert_to_tensor(["A", "A", "A"], dtype="string"),
        "sales": ops.convert_to_tensor([10.0, 11.0, 12.0], dtype="float32"),
    }
    _ = layer(train, training=True)
    config = layer.get_config()
    assert config["target_features"] == ["sales"]
    assert config["min_samples"] == 2
    assert "store" in config["group_configs"]
    assert config["combination_stats"]

    restored = GroupStatisticsLayer.from_config(config)
    assert restored.target_features == ["sales"]
    assert restored.min_samples == 2
    assert "store" in restored.group_configs

    # Trigger build + restore of combination weights.
    out_orig = layer(train, training=False)
    out_restored = restored(train, training=False)
    np.testing.assert_allclose(
        _as_numpy(out_orig["sales"]["anomaly_score"]),
        _as_numpy(out_restored["sales"]["anomaly_score"]),
        rtol=1e-5,
    )


def test_keras_model_roundtrip(tmp_path: Path, layer: GroupStatisticsLayer) -> None:
    """Functional model .keras save/load preserves scoring behavior."""
    train = {
        "store_id": ops.convert_to_tensor(["A", "A", "A", "A"], dtype="string"),
        "sales": ops.convert_to_tensor([10.0, 11.0, 12.0, 100.0], dtype="float32"),
    }
    _ = layer(train, training=True)

    inputs = {
        "store_id": keras.Input(shape=(1,), dtype="string", name="store_id"),
        "sales": keras.Input(shape=(1,), dtype="float32", name="sales"),
    }
    # Rebuild a fresh layer from config so combination_stats are constructor args.
    scoring_layer = GroupStatisticsLayer.from_config(layer.get_config())
    outputs = scoring_layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Materialize weights.
    _ = model(train, training=False)

    path = str(tmp_path / "group_stats.keras")
    model.save(path)
    loaded = keras.models.load_model(path)

    orig = model(train, training=False)
    loaded_out = loaded(train, training=False)
    np.testing.assert_allclose(
        _as_numpy(orig["sales"]["anomaly_score"]),
        _as_numpy(loaded_out["sales"]["anomaly_score"]),
        rtol=1e-4,
        atol=1e-4,
    )


def test_stratified_model_still_works() -> None:
    """KerasStratifiedAnomalyModel still wires GroupStatisticsLayer correctly."""
    from kerasfactory.models.KerasStratifiedAnomalyModel import (
        KerasStratifiedAnomalyModel,
    )

    model = KerasStratifiedAnomalyModel(
        feature_space={"store_id": "string", "sales": "float"},
        key_features=["store_id"],
        target_features=["sales"],
        min_samples=2,
        threshold=2.0,
    )
    inputs = {
        "store_id": ops.convert_to_tensor(
            [["A"], ["A"], ["A"], ["A"]],
            dtype="string",
        ),
        "sales": ops.convert_to_tensor(
            [[10.0], [11.0], [12.0], [100.0]],
            dtype="float32",
        ),
    }
    _ = model(inputs, training=True)
    outputs = model(inputs, training=False)
    assert "sales_anomaly_score" in outputs
    assert outputs["sales_anomaly_score"].shape == (4, 1)
    scores = _as_numpy(outputs["sales_anomaly_score"]).reshape(-1)
    assert scores[3] > scores[0]
