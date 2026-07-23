"""Global anomaly merge layer for combining feature-level anomaly outputs."""

from __future__ import annotations

from typing import Any

from keras import KerasTensor, ops
from keras.saving import register_keras_serializable
from loguru import logger

from kerasfactory.layers._base_layer import BaseLayer


@register_keras_serializable(package="kerasfactory.layers")
class GlobalAnomalyMergeLayer(BaseLayer):
    """Merges feature-level anomaly results into row-level predictions.

    Combines per-feature ``proba``, ``score``, ``anomaly``, and ``reason``
    dictionaries into global outputs using a ``max`` or ``mean`` strategy.

    Args:
        merge_strategy: How to combine feature scores (``"max"`` or ``"mean"``).
        format_details: Whether to include score details in reason strings.
        name: Optional layer name.
        **kwargs: Additional ``BaseLayer`` arguments.

    Example:
        ```python
        layer = GlobalAnomalyMergeLayer(merge_strategy="max")
        outputs = layer({
            "temp": {"proba": [[30.0]], "score": [[1.5]], "anomaly": [[False]],
                     "reason": ["ok"]},
            "color": {"proba": [[100.0]], "score": [[2.0]], "anomaly": [[True]],
                      "reason": ["bad"]},
        })
        ```
    """

    def __init__(
        self,
        merge_strategy: str = "max",
        format_details: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the layer.

        Args:
            merge_strategy: ``"max"`` or ``"mean"``.
            format_details: Whether to format detailed reason strings.
            name: Optional layer name.
            **kwargs: Additional layer arguments.
        """
        self._merge_strategy = merge_strategy
        self._format_details = format_details
        self._validate_params()

        self.merge_strategy = self._merge_strategy
        self.format_details = self._format_details

        super().__init__(name=name, **kwargs)

        score_strings = [f"{i / 100:.2f}" for i in range(10001)]
        self.score_lookup = ops.convert_to_tensor(score_strings, dtype="string")
        logger.debug(
            f"GlobalAnomalyMergeLayer strategy={self.merge_strategy}, "
            f"format_details={self.format_details}",
        )

    def _validate_params(self) -> None:
        """Validate merge strategy.

        Raises:
            ValueError: If merge_strategy is not supported.
        """
        if self._merge_strategy not in {"max", "mean"}:
            raise ValueError(
                f"Invalid merge_strategy: {self._merge_strategy}. "
                "Must be 'max' or 'mean'.",
            )

    def call(
        self,
        inputs: dict[str, dict[str, Any] | Any],
        training: bool | None = None,
    ) -> dict[str, KerasTensor]:
        """Combine feature-level anomaly results into global predictions.

        Args:
            inputs: Mapping of feature names to anomaly output dictionaries.
            training: Unused; kept for Keras call signature compatibility.

        Returns:
            Dictionary with ``probability``, ``score``, ``is_anomaly``,
            ``reason``, and ``details``.
        """
        del training

        probas: list[Any] = []
        scores: list[Any] = []
        anomalies: list[Any] = []
        reasons: list[Any] = []
        feature_names: list[str] = []

        first_output = next(iter(inputs.values()))
        first_tensor = None
        if isinstance(first_output, dict):
            for key in ("probability", "proba", "score", "is_anomaly", "anomaly"):
                if key in first_output:
                    first_tensor = first_output[key]
                    break
        else:
            first_tensor = first_output
        batch_size = ops.shape(first_tensor)[0]
        default_shape = ops.zeros((batch_size, 1), dtype="float32")

        for feat_name, feat_output in inputs.items():
            # Skip aggregation metadata / flat aggregation tensors before Keras
            # structure walking issues and before treating them as features.
            if feat_name == "aggregation_levels" or feat_name.startswith(
                "aggregation_",
            ):
                continue

            if isinstance(feat_output, dict):
                proba = feat_output.get(
                    "proba",
                    feat_output.get("probability", ops.zeros_like(default_shape)),
                )
                score = feat_output.get("score", ops.zeros_like(default_shape))
                anomaly = feat_output.get(
                    "anomaly",
                    feat_output.get(
                        "is_anomaly",
                        ops.zeros_like(default_shape, dtype="bool"),
                    ),
                )
                reason = feat_output.get(
                    "reason",
                    ops.broadcast_to(
                        ops.convert_to_tensor("No anomalies detected", dtype="string"),
                        (batch_size, 1),
                    ),
                )
            else:
                proba = inputs.get(f"{feat_name}_proba", ops.zeros_like(default_shape))
                score = inputs.get(f"{feat_name}_score", ops.zeros_like(default_shape))
                anomaly = inputs.get(
                    f"{feat_name}_anomaly",
                    ops.zeros_like(default_shape, dtype="bool"),
                )
                reason = inputs.get(
                    f"{feat_name}_reason",
                    ops.broadcast_to(
                        ops.convert_to_tensor(
                            f"No anomalies in {feat_name}",
                            dtype="string",
                        ),
                        (batch_size, 1),
                    ),
                )

            probas.append(proba)
            scores.append(score)
            anomalies.append(anomaly)
            reasons.append(reason)
            feature_names.append(feat_name)

        normalized_probas: list[Any] = []
        normalized_scores: list[Any] = []
        normalized_anomalies: list[Any] = []

        for proba, score, anomaly in zip(probas, scores, anomalies, strict=True):
            if ops.ndim(proba) > 2:
                proba = (
                    proba[:, 0, :]
                    if ops.shape(proba)[1] > 0
                    else ops.zeros((ops.shape(proba)[0], 1), dtype=proba.dtype)
                )
            if proba.dtype != "float32":
                proba = ops.cast(proba, dtype="float32")
            normalized_probas.append(proba)

            if ops.ndim(score) > 2:
                score = (
                    score[:, 0, :]
                    if ops.shape(score)[1] > 0
                    else ops.zeros((ops.shape(score)[0], 1), dtype=score.dtype)
                )
            if score.dtype != "float32":
                score = ops.cast(score, dtype="float32")
            normalized_scores.append(score)

            if ops.ndim(anomaly) > 2:
                anomaly = (
                    anomaly[:, 0, :]
                    if ops.shape(anomaly)[1] > 0
                    else ops.zeros((ops.shape(anomaly)[0], 1), dtype=anomaly.dtype)
                )
            if anomaly.dtype == "string":
                anomaly = ops.zeros(ops.shape(anomaly), dtype="bool")
            elif anomaly.dtype != "bool":
                anomaly = ops.cast(anomaly, dtype="bool")
            normalized_anomalies.append(anomaly)

        probas_t = ops.stack(normalized_probas, axis=1)
        scores_t = ops.stack(normalized_scores, axis=1)
        anomalies_t = ops.stack(normalized_anomalies, axis=1)

        if self.merge_strategy == "max":
            global_score = ops.max(scores_t, axis=1)
            global_proba = ops.max(probas_t, axis=1)
        else:
            global_score = ops.mean(scores_t, axis=1)
            global_proba = ops.mean(probas_t, axis=1)

        global_anomaly = ops.any(anomalies_t, axis=1)

        batch_size = ops.shape(anomalies_t)[0]

        no_anomalies = ops.reshape(
            ops.tile(
                ops.reshape(
                    ops.convert_to_tensor(["No anomalies detected"], dtype="string"),
                    (1, 1),
                ),
                [batch_size, 1],
            ),
            [batch_size],
        )

        feature_order = sorted(
            range(len(feature_names)),
            key=lambda i: feature_names[i],
        )

        combined_reasons = ops.reshape(
            ops.tile(
                ops.reshape(ops.convert_to_tensor([""], dtype="string"), (1, 1)),
                [batch_size, 1],
            ),
            [batch_size],
        )

        for i in range(len(feature_names)):
            feat_idx = feature_order[i]
            feat_name = feature_names[feat_idx]
            reason = ops.squeeze(reasons[feat_idx])
            score = scores_t[:, feat_idx, :]
            is_anomaly = ops.squeeze(anomalies_t[:, feat_idx, :])

            feat_str = ops.convert_to_tensor(feat_name + ": ")
            semicolon = ops.convert_to_tensor("; ")

            if ops.dtype(reason) != "string":
                reason = ops.convert_to_tensor("")

            if self.format_details:
                score_idx = ops.clip(
                    ops.cast(ops.squeeze(score) * 100, dtype="int32"),
                    0,
                    10000,
                )
                score_str = ops.take(self.score_lookup, score_idx)
                # Preserve StatsAnomaly formatting quirk (b'X.XX' in reason text).
                score_str = (
                    ops.convert_to_tensor("b'") + score_str + ops.convert_to_tensor("'")
                )
                score_prefix = ops.convert_to_tensor(" (score: ")
                score_suffix = ops.convert_to_tensor(")")
                formatted = feat_str + reason + score_prefix + score_str + score_suffix
            else:
                formatted = feat_str + reason

            first_reason = formatted
            appended_reason = combined_reasons + semicolon + formatted

            combined_reasons = ops.where(
                is_anomaly,
                ops.where(
                    ops.equal(combined_reasons, ""),
                    first_reason,
                    appended_reason,
                ),
                combined_reasons,
            )

        global_reason = ops.where(
            ops.equal(combined_reasons, ""),
            no_anomalies,
            combined_reasons,
        )

        global_proba = ops.reshape(global_proba, (-1, 1))
        global_score = ops.reshape(global_score, (-1, 1))
        global_anomaly = ops.reshape(global_anomaly, (-1, 1))
        global_reason = ops.convert_to_tensor(global_reason, dtype="string")

        return {
            "probability": global_proba,
            "score": global_score,
            "is_anomaly": global_anomaly,
            "reason": global_reason,
            "details": ops.convert_to_tensor(global_reason, dtype="string"),
        }

    def compute_output_spec(self, inputs_spec: Any) -> dict[str, KerasTensor]:
        """Compute output tensor specs.

        Args:
            inputs_spec: Unused input specification.

        Returns:
            Mapping of output names to KerasTensor specs.
        """
        del inputs_spec
        return {
            "probability": KerasTensor(shape=(None, 1), dtype="float32"),
            "score": KerasTensor(shape=(None, 1), dtype="float32"),
            "is_anomaly": KerasTensor(shape=(None, 1), dtype="bool"),
            "reason": KerasTensor(shape=(None,), dtype="string"),
            "details": KerasTensor(shape=(None,), dtype="string"),
        }

    def get_config(self) -> dict[str, Any]:
        """Return layer configuration for serialization.

        Returns:
            Dictionary of constructor parameters.
        """
        config = super().get_config()
        config.update(
            {
                "merge_strategy": self.merge_strategy,
                "format_details": self.format_details,
            },
        )
        return config
