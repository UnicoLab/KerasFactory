"""Group-stratified online statistics and z-score anomaly detection.

Extracted from StatsAnomaly ``GroupStatisticsLayer`` (``keras_stratified_model``)
and adapted to KerasFactory ``BaseLayer`` conventions.

Numeric aggregation uses Keras 3 ``ops`` (Welford online mean/std) for fixed
global / schema-level weights. Per-combination statistics use Python/numpy
state because Keras 3 forbids ``add_weight`` after ``build()``. String key
joining, ``unique``, and ``boolean_mask`` require TensorFlow APIs that Keras
ops do not yet provide for string tensors; those call sites are isolated and
documented below.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from keras import KerasTensor, ops
from keras.saving import register_keras_serializable
from loguru import logger

# TensorFlow is required for string join / unique / boolean_mask patterns that
# Keras 3 ops lack for string tensors (same constraint as other KF string paths).
import tensorflow as tf

from kerasfactory.layers._base_layer import BaseLayer

GroupConfig = dict[str, Any]
GroupConfigs = dict[str, GroupConfig]


def _sanitize_weight_name(raw: str, *, max_len: int = 80) -> str:
    """Sanitize a dynamic weight name for ``add_weight``.

    Args:
        raw: Raw combination / group identifier.
        max_len: Maximum length of the sanitized name.

    Returns:
        Safe weight name fragment.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
    if not cleaned:
        cleaned = "key"
    return cleaned[:max_len]


@register_keras_serializable(package="kerasfactory.layers")
class GroupStatisticsLayer(BaseLayer):
    """Maintain per-combination online statistics and detect z-score anomalies.

    For each named group configuration, categorical key features are joined into
    a per-sample combination key (``"|"``-separated). The layer tracks running
    mean / std / count of ``target_features`` **per unique combination** using
    Welford's online algorithm, plus global statistics as a fallback.

    During training, statistics are updated for every unique key present in the
    batch. During inference, each sample is scored with its combination's
    absolute z-scores when that combination has at least ``min_samples``
    observations; otherwise global statistics are used
    (``fallback_strategy="global"``).

    When multiple group configs are defined, samples are assigned to the first
    matching config in insertion order (``processed_mask``), matching the
    original StatsAnomaly layer.

    Args:
        group_configs: Mapping of group name → config. Each config must include
            ``keys`` (list of stratification feature names) and may include
            ``threshold`` (float, default ``3.0``).
        target_features: Numerical feature names to score for anomalies.
        min_samples: Minimum samples required before combination stats are
            trusted. Defaults to ``5``.
        fallback_strategy: Strategy when a combination is under-sampled.
            Currently only ``"global"`` is supported.
        epsilon: Numerical stability constant for division / std. Defaults to
            ``1e-10``.
        name: Optional layer name.
        combination_stats: Optional restored combination state from
            ``get_config``.
        **kwargs: Additional ``BaseLayer`` / Keras layer arguments.

    Input:
        Dictionary mapping feature names to tensors of shape ``(batch,)`` or
        ``(batch, 1)``. Must include every name in ``target_features`` and every
        key listed under ``group_configs[*]["keys"]``.

    Output:
        Nested dictionary ``{feature: {anomaly_score, is_anomaly, reason}}``
        where each tensor has shape ``(batch_size,)``.

    Note:
        Combination statistics are stored as Python/numpy state (not layer
        weights) so they can grow dynamically under Keras 3. Prefer eager
        ``call(..., training=True)`` when fitting. State is serialized via
        ``get_config`` / ``from_config``.

    Example:
        ```python
        import keras
        from kerasfactory.layers.GroupStatisticsLayer import GroupStatisticsLayer

        layer = GroupStatisticsLayer(
            group_configs={
                "product_store": {
                    "keys": ["product_category", "store_location"],
                    "threshold": 3.0,
                }
            },
            target_features=["sales", "inventory"],
            min_samples=5,
        )
        outputs = layer(
            {
                "product_category": keras.ops.convert_to_tensor(
                    ["electronics", "electronics"],
                    dtype="string",
                ),
                "store_location": keras.ops.convert_to_tensor(
                    ["north", "north"],
                    dtype="string",
                ),
                "sales": keras.ops.convert_to_tensor([1000.0, 5000.0]),
                "inventory": keras.ops.convert_to_tensor([100.0, 120.0]),
            },
            training=True,
        )
        ```
    """

    def __init__(
        self,
        group_configs: GroupConfigs,
        target_features: list[str],
        min_samples: int = 5,
        fallback_strategy: str = "global",
        epsilon: float = 1e-10,
        name: str | None = None,
        combination_stats: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the layer.

        Args:
            group_configs: Group configuration mapping.
            target_features: Numerical target feature names.
            min_samples: Minimum samples for valid combination statistics.
            fallback_strategy: Under-sampled combination fallback strategy.
            epsilon: Numerical stability constant.
            name: Optional layer name.
            combination_stats: Optional restored combination state.
            **kwargs: Additional layer arguments.
        """
        self._group_configs = group_configs
        self._target_features = list(target_features)
        self._min_samples = int(min_samples)
        self._fallback_strategy = fallback_strategy
        self._epsilon = float(epsilon)
        self._pending_combination_stats = combination_stats or {}

        self._validate_params()

        # Public attributes before super().__init__ (BaseLayer logs get_config).
        self.group_configs = self._group_configs
        self.target_features = self._target_features
        self.min_samples = self._min_samples
        self.fallback_strategy = self._fallback_strategy
        self.epsilon = self._epsilon

        # Schema-level aggregates (one weight set per group_config name).
        self.group_means: dict[str, Any] = {}
        self.group_stds: dict[str, Any] = {}
        self.group_counts: dict[str, Any] = {}
        # Per unique key combination — Python/numpy state (not add_weight).
        self.combination_stats: dict[str, dict[str, Any]] = {}
        for combo_id, stats in self._pending_combination_stats.items():
            self.combination_stats[combo_id] = {
                "mean": np.asarray(stats["mean"], dtype=np.float32),
                "std": np.asarray(stats["std"], dtype=np.float32),
                "count": float(stats["count"]),
            }
        self.global_mean = None
        self.global_std = None
        self.global_count = None

        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters.

        Raises:
            ValueError: If configs, features, or strategy are invalid.
        """
        if not isinstance(self._group_configs, dict) or not self._group_configs:
            raise ValueError("group_configs must be a non-empty dictionary")
        if not self._target_features:
            raise ValueError("target_features must be a non-empty list of strings")
        if any(not isinstance(f, str) or not f for f in self._target_features):
            raise ValueError("target_features entries must be non-empty strings")
        if self._min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self._min_samples}")
        if self._fallback_strategy != "global":
            raise ValueError(
                "fallback_strategy must be 'global' "
                f"(got {self._fallback_strategy!r})",
            )
        if self._epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self._epsilon}")

        for group_key, config in self._group_configs.items():
            if not isinstance(config, dict):
                raise ValueError(f"group_configs[{group_key!r}] must be a dict")
            keys = config.get("keys")
            if not isinstance(keys, list) or not keys:
                raise ValueError(
                    f"group_configs[{group_key!r}]['keys'] must be a non-empty list",
                )
            threshold = config.get("threshold", 3.0)
            if not isinstance(threshold, int | float) or float(threshold) <= 0:
                raise ValueError(
                    f"group_configs[{group_key!r}]['threshold'] must be positive",
                )

    def build(
        self,
        input_shape: dict[str, Any] | tuple[Any, ...] | None = None,
    ) -> None:
        """Create non-trainable global / schema statistic weights.

        Args:
            input_shape: Dict of per-feature shapes, or unused placeholder.
        """
        if self.built and self.global_mean is not None:
            return

        num_features = len(self.target_features)

        for group_key in self.group_configs:
            safe_name = _sanitize_weight_name(str(group_key))
            self.group_means[group_key] = self.add_weight(
                name=f"mean_{safe_name}",
                shape=(num_features,),
                initializer="zeros",
                trainable=False,
            )
            self.group_stds[group_key] = self.add_weight(
                name=f"std_{safe_name}",
                shape=(num_features,),
                initializer="ones",
                trainable=False,
            )
            self.group_counts[group_key] = self.add_weight(
                name=f"count_{safe_name}",
                shape=(),
                initializer="zeros",
                trainable=False,
            )

        self.global_mean = self.add_weight(
            name="global_mean",
            shape=(num_features,),
            initializer="zeros",
            trainable=False,
        )
        self.global_std = self.add_weight(
            name="global_std",
            shape=(num_features,),
            initializer="ones",
            trainable=False,
        )
        self.global_count = self.add_weight(
            name="global_count",
            shape=(),
            initializer="zeros",
            trainable=False,
        )

        logger.debug(
            "Building GroupStatisticsLayer "
            f"(groups={list(self.group_configs)}, "
            f"targets={self.target_features}, min_samples={self.min_samples}, "
            f"combos={len(self.combination_stats)})",
        )
        super().build(input_shape if input_shape is not None else ())

    def compute_output_shape(
        self,
        input_shape: dict[str, Any],
    ) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Compute nested output shapes.

        Args:
            input_shape: Mapping of feature name → shape.

        Returns:
            Nested shape mapping matching :meth:`call` outputs.
        """
        first_shape = next(iter(input_shape.values()))
        batch_size = first_shape[0] if first_shape is not None else None
        per_feature = {
            "anomaly_score": (batch_size,),
            "is_anomaly": (batch_size,),
            "reason": (batch_size,),
        }
        return {feature: dict(per_feature) for feature in self.target_features}

    def call(
        self,
        inputs: dict[str, KerasTensor],
        training: bool | None = None,
    ) -> dict[str, dict[str, KerasTensor]]:
        """Update statistics (train) and compute per-feature anomaly outputs.

        Args:
            inputs: Feature-name → tensor dictionary.
            training: If ``True``, update online statistics before scoring.

        Returns:
            Nested anomaly outputs per target feature.
        """
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError("inputs must be a non-empty dictionary of tensors")

        missing_targets = [f for f in self.target_features if f not in inputs]
        if missing_targets:
            raise ValueError(f"inputs missing target_features: {missing_targets}")

        for group_key, config in self.group_configs.items():
            missing_keys = [k for k in config["keys"] if k not in inputs]
            if missing_keys:
                raise ValueError(
                    f"inputs missing keys for group {group_key!r}: {missing_keys}",
                )

        if not self.built or self.global_mean is None:
            self.build({name: getattr(t, "shape", None) for name, t in inputs.items()})

        processed = self._preprocess_inputs(inputs)
        batch_size = ops.shape(next(iter(processed.values())))[0]
        target_values = ops.stack(
            [ops.cast(processed[name], "float32") for name in self.target_features],
            axis=1,
        )
        target_values = ops.reshape(
            target_values,
            (batch_size, len(self.target_features)),
        )

        group_key_tensors: list[tuple[str, Any]] = []
        for group_key, config in self.group_configs.items():
            sample_keys = self._join_key_features(processed, config["keys"], batch_size)
            group_key_tensors.append((group_key, sample_keys))

        if training:
            self._update_statistics(group_key_tensors, target_values, batch_size)

        return self._compute_anomaly_scores(
            group_key_tensors,
            target_values,
            batch_size,
        )

    def _preprocess_inputs(
        self,
        inputs: dict[str, KerasTensor],
    ) -> dict[str, KerasTensor]:
        """Flatten feature tensors to shape ``(batch_size,)``.

        Args:
            inputs: Raw feature dictionary.

        Returns:
            Dictionary of rank-1 tensors.
        """
        first_tensor = next(iter(inputs.values()))
        batch_size = ops.shape(first_tensor)[0]
        return {
            name: ops.reshape(tensor, (batch_size,)) for name, tensor in inputs.items()
        }

    def _join_key_features(
        self,
        processed: dict[str, KerasTensor],
        key_names: list[str],
        batch_size: Any,
    ) -> Any:
        """Join stratification features into per-sample combination keys.

        Uses ``tf.strings.join`` because Keras ops lack a string-join for
        heterogeneous key dtypes.

        Args:
            processed: Rank-1 feature tensors.
            key_names: Stratification feature names.
            batch_size: Batch size tensor / int.

        Returns:
            String tensor of shape ``(batch_size,)``.
        """
        parts: list[Any] = []
        for name in key_names:
            tensor = tf.convert_to_tensor(processed[name])
            tensor = tf.reshape(tensor, [batch_size])
            if tensor.dtype != tf.string:
                tensor = tf.strings.as_string(tensor)
            parts.append(tensor)
        return tf.strings.join(parts, separator="|")

    def _combination_id(self, group_key: str, sample_key: Any) -> str:
        """Build a stable combination id from group name and sample key.

        Args:
            group_key: Group config name.
            sample_key: Scalar string key (tensor or Python / bytes).

        Returns:
            Combination identifier string.
        """
        if hasattr(sample_key, "numpy"):
            sample_key = sample_key.numpy()
        if isinstance(sample_key, bytes):
            sample_key = sample_key.decode("utf-8")
        return f"{group_key}::{sample_key}"

    def _ensure_combination_stats(self, combo_id: str) -> dict[str, Any]:
        """Return (creating if needed) numpy stats for a combination.

        Args:
            combo_id: Combination identifier.

        Returns:
            Mutable dict with ``mean``, ``std``, and ``count``.
        """
        if combo_id not in self.combination_stats:
            n = len(self.target_features)
            self.combination_stats[combo_id] = {
                "mean": np.zeros(n, dtype=np.float32),
                "std": np.ones(n, dtype=np.float32),
                "count": 0.0,
            }
        return self.combination_stats[combo_id]

    def _welford_update_numpy(
        self,
        stats: dict[str, Any],
        values: np.ndarray,
    ) -> None:
        """Welford-merge a numpy batch into combination stats in-place.

        Args:
            stats: Mutable ``{mean, std, count}`` dict.
            values: Array of shape ``(batch, num_features)``.
        """
        if values.size == 0:
            return
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, len(self.target_features))
        n_b = float(values.shape[0])
        batch_mean = values.mean(axis=0)
        batch_m2 = np.sum((values - batch_mean) ** 2, axis=0)

        n_a = float(stats["count"])
        n = n_a + n_b
        delta = batch_mean - stats["mean"]
        new_mean = stats["mean"] + delta * (n_b / n)
        m2_a = (stats["std"] ** 2) * n_a
        new_m2 = m2_a + batch_m2 + (delta**2) * n_a * n_b / n
        new_std = np.sqrt(new_m2 / n + self.epsilon)

        stats["mean"] = new_mean.astype(np.float32)
        stats["std"] = new_std.astype(np.float32)
        stats["count"] = n

    def _welford_update(
        self,
        mean_var: Any,
        std_var: Any,
        count_var: Any,
        values: KerasTensor,
    ) -> None:
        """Merge a batch into Keras weight mean/std/count via Welford.

        Args:
            mean_var: Non-trainable mean weight ``(num_features,)``.
            std_var: Non-trainable std weight ``(num_features,)``.
            count_var: Non-trainable scalar count weight.
            values: Batch values ``(batch, num_features)``.
        """
        n_rows = int(tf.shape(values)[0].numpy())
        if n_rows == 0:
            return

        n_b = ops.cast(n_rows, "float32")
        batch_mean = ops.mean(values, axis=0)
        batch_m2 = ops.sum(ops.square(values - batch_mean), axis=0)

        n_a = count_var
        n = n_a + n_b
        delta = batch_mean - mean_var
        new_mean = mean_var + delta * (n_b / n)
        m2_a = ops.square(std_var) * n_a
        new_m2 = m2_a + batch_m2 + ops.square(delta) * n_a * n_b / n
        new_std = ops.sqrt(new_m2 / n + self.epsilon)

        mean_var.assign(new_mean)
        std_var.assign(new_std)
        count_var.assign(n)

    def _update_statistics(
        self,
        group_key_tensors: list[tuple[str, Any]],
        target_values: KerasTensor,
        batch_size: Any,
    ) -> None:
        """Update global, schema, and per-combination statistics during training.

        Args:
            group_key_tensors: ``(group_name, sample_keys)`` pairs.
            target_values: Stacked target values ``(batch, num_features)``.
            batch_size: Batch size tensor / int.
        """
        n_batch = int(ops.convert_to_numpy(ops.cast(batch_size, "int32")))
        if n_batch == 0:
            return

        self._welford_update(
            self.global_mean,
            self.global_std,
            self.global_count,
            target_values,
        )

        for group_key, sample_keys in group_key_tensors:
            self._welford_update(
                self.group_means[group_key],
                self.group_stds[group_key],
                self.group_counts[group_key],
                target_values,
            )

            # TF: unique + boolean_mask (Keras ops lack string unique/mask).
            unique_keys, _ = tf.unique(sample_keys)
            for uk in unique_keys.numpy().tolist():
                uk_tensor = tf.convert_to_tensor(uk, dtype=tf.string)
                mask = tf.equal(sample_keys, uk_tensor)
                group_values = tf.boolean_mask(target_values, mask)
                combo_id = self._combination_id(group_key, uk)
                stats = self._ensure_combination_stats(combo_id)
                self._welford_update_numpy(
                    stats,
                    np.asarray(group_values.numpy(), dtype=np.float32),
                )

    def _empty_outputs(self, batch_size: Any) -> dict[str, dict[str, KerasTensor]]:
        """Build empty / zero outputs for edge-case batches.

        Args:
            batch_size: Batch size (may be 0).

        Returns:
            Nested anomaly dictionary with correct shapes.
        """
        default_reason = ops.convert_to_tensor("No anomaly detected", dtype="string")
        outputs: dict[str, dict[str, KerasTensor]] = {}
        for feature in self.target_features:
            outputs[feature] = {
                "anomaly_score": ops.zeros((batch_size,), dtype="float32"),
                "is_anomaly": ops.zeros((batch_size,), dtype="bool"),
                "reason": ops.broadcast_to(default_reason, (batch_size,)),
            }
        return outputs

    def _compute_anomaly_scores(
        self,
        group_key_tensors: list[tuple[str, Any]],
        target_values: KerasTensor,
        batch_size: Any,
    ) -> dict[str, dict[str, KerasTensor]]:
        """Compute absolute z-score anomaly outputs per target feature.

        Args:
            group_key_tensors: ``(group_name, sample_keys)`` pairs.
            target_values: Stacked target values ``(batch, num_features)``.
            batch_size: Batch size tensor / int.

        Returns:
            Nested anomaly dictionary.
        """
        n_batch = int(ops.convert_to_numpy(ops.cast(batch_size, "int32")))
        if n_batch == 0:
            return self._empty_outputs(batch_size)

        default_reason = ops.convert_to_tensor("No anomaly detected", dtype="string")
        global_z = ops.absolute(
            (target_values - self.global_mean) / (self.global_std + self.epsilon),
        )

        anomaly_scores: dict[str, dict[str, KerasTensor]] = {}
        for feature_idx, feature in enumerate(self.target_features):
            feature_scores = global_z[:, feature_idx]
            primary = next(iter(self.group_configs.values()))
            feature_threshold = float(primary.get("threshold", 3.0))
            feature_anomalies = ops.greater(feature_scores, feature_threshold)
            reasons = ops.broadcast_to(default_reason, (batch_size,))
            processed_mask = tf.zeros([batch_size], dtype=tf.bool)

            for group_key, sample_keys in group_key_tensors:
                group_threshold = float(
                    self.group_configs[group_key].get("threshold", 3.0),
                )
                unique_keys, _ = tf.unique(sample_keys)
                for uk in unique_keys.numpy().tolist():
                    uk_tensor = tf.convert_to_tensor(uk, dtype=tf.string)
                    group_mask = tf.equal(sample_keys, uk_tensor)
                    valid_mask = tf.logical_and(
                        group_mask,
                        tf.logical_not(processed_mask),
                    )
                    if not bool(tf.reduce_any(valid_mask).numpy()):
                        continue

                    combo_id = self._combination_id(group_key, uk)
                    stats = self._ensure_combination_stats(combo_id)
                    has_enough = stats["count"] >= float(self.min_samples)

                    if has_enough:
                        mean_i = ops.convert_to_tensor(
                            float(stats["mean"][feature_idx]),
                            dtype="float32",
                        )
                        std_i = ops.convert_to_tensor(
                            float(stats["std"][feature_idx]),
                            dtype="float32",
                        )
                    else:
                        mean_i = ops.take(self.global_mean, feature_idx)
                        std_i = ops.take(self.global_std, feature_idx)

                    values = target_values[:, feature_idx]
                    z_scores = ops.absolute((values - mean_i) / (std_i + self.epsilon))
                    group_anomalies = ops.greater(z_scores, group_threshold)

                    feature_scores = ops.where(valid_mask, z_scores, feature_scores)
                    feature_anomalies = ops.where(
                        valid_mask,
                        group_anomalies,
                        feature_anomalies,
                    )

                    uk_text = uk.decode() if isinstance(uk, bytes) else str(uk)
                    reason_group = ops.convert_to_tensor(
                        f"Anomaly in {feature}: value deviates from group "
                        f"{group_key} ({uk_text})",
                        dtype="string",
                    )
                    reason_global = ops.convert_to_tensor(
                        f"Anomaly in {feature}: value deviates from global statistics",
                        dtype="string",
                    )
                    anomaly_reason = reason_group if has_enough else reason_global
                    anomaly_mask = tf.logical_and(valid_mask, group_anomalies)
                    reasons = ops.where(anomaly_mask, anomaly_reason, reasons)
                    processed_mask = tf.logical_or(processed_mask, valid_mask)

            anomaly_scores[feature] = {
                "anomaly_score": feature_scores,
                "is_anomaly": feature_anomalies,
                "reason": reasons,
            }

        return anomaly_scores

    def get_combination_stats(self) -> dict[str, dict[str, Any]]:
        """Return a serializable snapshot of per-combination statistics.

        Returns:
            Mapping of combination id → ``{mean, std, count}``.
        """
        snapshot: dict[str, dict[str, Any]] = {}
        for combo_id, stats in self.combination_stats.items():
            snapshot[combo_id] = {
                "mean": np.asarray(stats["mean"], dtype=np.float32).tolist(),
                "std": np.asarray(stats["std"], dtype=np.float32).tolist(),
                "count": float(stats["count"]),
            }
        return snapshot

    def get_config(self) -> dict[str, Any]:
        """Return the layer configuration for serialization.

        Returns:
            Dictionary with constructor parameters and combination stats.
        """
        config = super().get_config()
        config.update(
            {
                "group_configs": self.group_configs,
                "target_features": self.target_features,
                "min_samples": self.min_samples,
                "fallback_strategy": self.fallback_strategy,
                "epsilon": self.epsilon,
                "combination_stats": self.get_combination_stats(),
            },
        )
        return config
