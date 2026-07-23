"""Statistical numerical anomaly detection with z-score, IQR, and MAD methods.

Merges the StatsAnomaly ``EnhancedNumericalAnomalyDetectionLayer`` and
``NumericalAnomalyDetectionLayer`` APIs into a single KerasFactory layer.
"""

from __future__ import annotations

from typing import Any, Literal

from keras import KerasTensor, ops
from keras.saving import register_keras_serializable
from loguru import logger

from kerasfactory.layers._base_layer import BaseLayer

StatisticalMethod = Literal["z-score", "iqr", "mad"]


@register_keras_serializable(package="kerasfactory.layers")
class StatisticalNumericalAnomalyDetection(BaseLayer):
    """Statistical anomaly detection for a numerical (scalar) feature.

    Supports multiple statistical methods:
        - ``z-score``: Absolute standardized deviation from the mean.
        - ``iqr``: Distance outside the interquartile fence (Tukey).
        - ``mad``: Absolute deviation from the median scaled by MAD.

    Statistics are stored as non-trainable weights so they persist across
    ``.keras`` save/load. Call :meth:`initialize_from_stats` after construction
    (or after loading) to set method-specific statistics.

    Args:
        method: Statistical method (``"z-score"``, ``"iqr"``, or ``"mad"``).
        threshold: Anomaly threshold for the chosen method. Defaults to ``2.0``.
        epsilon: Small constant for safe division. Defaults to ``1e-8``.
        name: Optional layer name.
        **kwargs: Additional ``BaseLayer`` / Keras layer arguments.

    Input shape:
        ``(batch_size, 1)`` or ``(batch_size,)`` — scalar numerical feature.

    Output shape:
        Dictionary with keys:
            - ``score``: ``(batch_size, 1)`` float32 deviation score.
            - ``proba``: ``(batch_size, 1)`` float32 anomaly probability in ``[0, 100]``.
            - ``threshold``: ``(1,)`` float32 numeric threshold value.
            - ``anomaly``: ``(batch_size, 1)`` bool flags.
            - ``reason``: ``(batch_size, 1)`` string explanations.
            - ``value``: ``(batch_size, 1)`` float32 input values.

    Example:
        ```python
        import keras
        from kerasfactory.layers.StatisticalNumericalAnomalyDetection import (
            StatisticalNumericalAnomalyDetection,
        )

        layer = StatisticalNumericalAnomalyDetection(method="z-score", threshold=2.0)
        layer.initialize_from_stats({"mean": 50.0, "std": 10.0})
        # Or simple API: layer.initialize_from_stats(mean=50.0, std=10.0)
        outputs = layer(keras.ops.convert_to_tensor([[55.0], [80.0]]))
        print(outputs["anomaly"])
        ```
    """

    def __init__(
        self,
        method: StatisticalMethod = "z-score",
        threshold: float = 2.0,
        epsilon: float = 1e-8,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the layer.

        Args:
            method: Statistical method to use.
            threshold: Anomaly detection threshold.
            epsilon: Small constant for safe division.
            name: Optional layer name.
            **kwargs: Additional layer arguments.
        """
        self._method = method
        self._threshold = float(threshold)
        self._epsilon = float(epsilon)

        self._validate_params()

        # Public attributes before super().__init__ (BaseLayer logs get_config).
        self.method = self._method
        self.threshold = self._threshold
        self.epsilon = self._epsilon

        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters.

        Raises:
            ValueError: If method, threshold, or epsilon is invalid.
        """
        valid_methods = {"z-score", "iqr", "mad"}
        if self._method not in valid_methods:
            raise ValueError(
                f"method must be one of {sorted(valid_methods)}, got {self._method!r}",
            )
        if self._threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self._threshold}")
        if self._epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self._epsilon}")

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        """Create non-trainable statistic weights.

        Args:
            input_shape: Input shape; expected ``(batch, 1)`` or ``(batch,)``.
        """
        self.mean = self.add_weight(
            name="mean",
            shape=(1,),
            initializer="zeros",
            trainable=False,
        )
        self.std = self.add_weight(
            name="std",
            shape=(1,),
            initializer="ones",
            trainable=False,
        )
        self.median = self.add_weight(
            name="median",
            shape=(1,),
            initializer="zeros",
            trainable=False,
        )
        self.q1 = self.add_weight(
            name="q1",
            shape=(1,),
            initializer="zeros",
            trainable=False,
        )
        self.q3 = self.add_weight(
            name="q3",
            shape=(1,),
            initializer="ones",
            trainable=False,
        )
        self.mad = self.add_weight(
            name="mad",
            shape=(1,),
            initializer="ones",
            trainable=False,
        )
        logger.debug(
            "Building StatisticalNumericalAnomalyDetection "
            f"(method={self.method}, threshold={self.threshold})",
        )
        super().build(input_shape)

    def initialize_from_stats(
        self,
        stats: dict[str, Any] | float | None = None,
        std: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize method-specific statistics.

        Compatible with both StatsAnomaly APIs:

        - Enhanced (dict): ``initialize_from_stats({"mean": 50.0, "std": 10.0})``
        - Simple (kwargs / positional): ``initialize_from_stats(mean=50.0, std=10.0)``
          or ``initialize_from_stats(50.0, 10.0)``

        Args:
            stats: Statistics dictionary, or positional mean when using the
                simple ``(mean, std)`` API.
            std: Standard deviation when using the simple positional API.
            **kwargs: Additional statistic keys (``mean``, ``std``, ``median``,
                ``q1``, ``q3``, ``mad``).

        Raises:
            ValueError: If required keys for the configured method are missing.
        """
        if not self.built:
            self.build((None, 1))

        if isinstance(stats, dict):
            data: dict[str, Any] = {**stats, **kwargs}
        elif isinstance(stats, int | float):
            if std is None and "std" not in kwargs:
                raise ValueError(
                    "When passing mean positionally, std must also be provided",
                )
            data = {
                "mean": float(stats),
                "std": float(std if std is not None else kwargs["std"]),
            }
            data.update({k: v for k, v in kwargs.items() if k != "std"})
        else:
            data = dict(kwargs)
            if std is not None:
                data["std"] = std

        if self.method == "z-score":
            if "mean" not in data or "std" not in data:
                raise ValueError("z-score method requires 'mean' and 'std' in stats")
            self.mean.assign([float(data["mean"])])
            self.std.assign([max(float(data["std"]), self.epsilon)])
            logger.info(
                "StatisticalNumericalAnomalyDetection (z-score) initialized with "
                f"mean={data['mean']}, std={data['std']}",
            )
        elif self.method == "iqr":
            if "q1" not in data or "q3" not in data:
                raise ValueError("iqr method requires 'q1' and 'q3' in stats")
            self.q1.assign([float(data["q1"])])
            self.q3.assign([float(data["q3"])])
            logger.info(
                "StatisticalNumericalAnomalyDetection (iqr) initialized with "
                f"q1={data['q1']}, q3={data['q3']}",
            )
        elif self.method == "mad":
            if "median" not in data or "mad" not in data:
                raise ValueError("mad method requires 'median' and 'mad' in stats")
            self.median.assign([float(data["median"])])
            self.mad.assign([max(float(data["mad"]), self.epsilon)])
            logger.info(
                "StatisticalNumericalAnomalyDetection (mad) initialized with "
                f"median={data['median']}, mad={data['mad']}",
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> dict[str, tuple[int | None, ...]]:
        """Compute output shapes for each dictionary key.

        Args:
            input_shape: Input shape tuple.

        Returns:
            Mapping of output names to shapes.
        """
        batch_size = input_shape[0]
        return {
            "score": (batch_size, 1),
            "proba": (batch_size, 1),
            "threshold": (1,),
            "anomaly": (batch_size, 1),
            "reason": (batch_size, 1),
            "value": (batch_size, 1),
        }

    def call(
        self,
        inputs: KerasTensor,
        training: bool | None = None,
    ) -> dict[str, KerasTensor]:
        """Detect anomalies in the input data.

        Args:
            inputs: Input tensor of shape ``(batch_size, 1)`` or ``(batch_size,)``.
            training: Unused; present for Keras call signature compatibility.

        Returns:
            Dictionary with ``score``, ``proba``, ``threshold``, ``anomaly``,
            ``reason``, and ``value``.
        """
        del training  # Unused.
        x = ops.cast(ops.convert_to_tensor(inputs), dtype="float32")
        if ops.ndim(x) == 1:
            x = ops.expand_dims(x, -1)

        if self.method == "z-score":
            safe_std = ops.maximum(self.std, self.epsilon)
            score = ops.absolute(x - self.mean) / safe_std
            anomaly_reason = ops.convert_to_tensor(
                f"Statistical anomaly: |z-score| exceeds {self.threshold}",
                dtype="string",
            )
            normal_reason = ops.convert_to_tensor(
                "Value within statistical range",
                dtype="string",
            )
        elif self.method == "iqr":
            iqr = ops.maximum(self.q3 - self.q1, self.epsilon)
            lower_bound = self.q1 - self.threshold * iqr
            upper_bound = self.q3 + self.threshold * iqr
            lower_score = ops.maximum(0.0, (lower_bound - x) / iqr)
            upper_score = ops.maximum(0.0, (x - upper_bound) / iqr)
            score = ops.maximum(lower_score, upper_score)
            anomaly_reason = ops.convert_to_tensor(
                f"Value outside {self.threshold} × IQR from quartiles",
                dtype="string",
            )
            normal_reason = ops.convert_to_tensor(
                "Value within IQR fence",
                dtype="string",
            )
        elif self.method == "mad":
            safe_mad = ops.maximum(self.mad, self.epsilon)
            score = ops.absolute(x - self.median) / safe_mad
            anomaly_reason = ops.convert_to_tensor(
                f"Statistical anomaly: MAD score exceeds {self.threshold}",
                dtype="string",
            )
            normal_reason = ops.convert_to_tensor(
                "Value within MAD range",
                dtype="string",
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        is_anomaly = score > self.threshold

        # Sigmoid-style mapping of score relative to threshold → [0, 100].
        proba = 100.0 * (1.0 - 1.0 / (1.0 + ops.exp(score - self.threshold)))

        reason = ops.where(is_anomaly, anomaly_reason, normal_reason)
        threshold_tensor = ops.convert_to_tensor([self.threshold], dtype="float32")

        return {
            "score": score,
            "proba": proba,
            "threshold": threshold_tensor,
            "anomaly": is_anomaly,
            "reason": reason,
            "value": x,
        }

    def get_config(self) -> dict[str, Any]:
        """Return the layer configuration for serialization.

        Returns:
            Dictionary with constructor parameters.
        """
        config = super().get_config()
        config.update(
            {
                "method": self.method,
                "threshold": self.threshold,
                "epsilon": self.epsilon,
            },
        )
        return config
