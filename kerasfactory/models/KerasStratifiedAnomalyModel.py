"""Keras stratified anomaly detection model.

Ported from StatsAnomaly ``keras_stratified_model.py``. Uses
``kerasfactory.layers.GroupStatisticsLayer`` for stratified z-score scoring.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import keras
import numpy as np
from keras.saving import register_keras_serializable
from loguru import logger

from kerasfactory.layers import GroupStatisticsLayer

# Prefer Keras dtypes; fall back to tensorflow only for Dataset typing in fit/predict.
try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None  # type: ignore[assignment]


@register_keras_serializable(package="kerasfactory.models")
class KerasStratifiedAnomalyModel(keras.Model):
    """A Keras-based implementation of the stratified anomaly detection model.

    This model detects anomalies in time series data based on statistical properties
    of groups defined by categorical features. It is fully serializable and compatible
    with TensorFlow Serving, allowing for direct inference on raw data in production environments.

    The model architecture consists of:
    1. Input layers for key features (categorical) and target features (numerical)
    2. A GroupStatisticsLayer that computes statistics for each group and detects anomalies
    3. Output layers for anomaly scores, flags, and reasons

    During training, the model updates the statistics for each group using Welford's online
    algorithm for numerical stability. During inference, the model uses these statistics to
    compute anomaly scores and flags based on z-scores (number of standard deviations from the mean).

    The model can be saved and loaded using the save_model and load_model methods, which ensure
    compatibility with Keras 3 and TensorFlow Serving. It also provides methods for generating
    detailed reports about detected anomalies.

    Args:
        feature_space: Dictionary mapping feature names to their data types.
        key_features: List of features to use for stratification. These should be categorical features.
        target_features: List of features to detect anomalies in. These should be numerical features.
        min_samples: Minimum number of samples required to compute statistics for a group.
            Groups with fewer samples will use the fallback strategy.
        fallback_strategy: Strategy to use when a group has insufficient samples. Currently
            only 'global' is supported, which uses global statistics computed across all groups.
        threshold: Threshold for anomaly detection, specified as the number of standard deviations
            from the mean. Default is 3.0 (3-sigma rule).
        name: Name of the model. Used for serialization and debugging.
    """

    def __init__(
        self,
        feature_space: dict[str, str],
        key_features: list[str],
        target_features: list[str],
        min_samples: int = 5,
        fallback_strategy: str = "global",
        threshold: float = 3.0,
        name: str = "stratified_anomaly_model",
        **kwargs,
    ) -> None:
        """Initialize the stratified anomaly model.

        Args:
            feature_space: Mapping of feature name to dtype string
                (``"string"`` or ``"float"``).
            key_features: Categorical features used for stratification.
            target_features: Numerical features to score for anomalies.
            min_samples: Minimum samples before combination stats are trusted.
            fallback_strategy: Strategy when a combination is under-sampled.
            threshold: Z-score threshold for anomaly flags.
            name: Model name.
            **kwargs: Additional ``keras.Model`` arguments.
        """
        super().__init__(name=name, **kwargs)
        self.feature_space = feature_space
        self.key_features = key_features
        self.target_features = target_features
        self.min_samples = min_samples
        self.fallback_strategy = fallback_strategy
        self.threshold = threshold

        # Define input layers
        self.input_layers = {}
        for feature, dtype in feature_space.items():
            if dtype == "string":
                self.input_layers[feature] = keras.layers.Input(
                    shape=(1,),
                    name=feature,
                    dtype="string",
                )
            else:
                self.input_layers[feature] = keras.layers.Input(
                    shape=(1,),
                    name=feature,
                    dtype="float32",
                )

        # Generate all possible group combinations
        self.group_configs = self._generate_group_configs()

        # Create the statistics layer
        self.statistics_layer = GroupStatisticsLayer(
            group_configs=self.group_configs,
            target_features=target_features,
            min_samples=min_samples,
            fallback_strategy=fallback_strategy,
        )

        # Define output layers
        self.output_layers = {}
        for feature in target_features:
            self.output_layers[f"{feature}_anomaly_score"] = keras.layers.Dense(
                1,
                name=f"{feature}_anomaly_score",
            )
            self.output_layers[f"{feature}_is_anomaly"] = keras.layers.Dense(
                1,
                activation="sigmoid",
                name=f"{feature}_is_anomaly",
            )

    def _generate_group_configs(self) -> dict[str, dict[str, Any]]:
        """Generate configurations for all possible group combinations.

        Returns:
            Dictionary mapping group keys to their configurations.
        """
        # For simplicity, we'll just create a single group config for all key features
        group_key = "_".join(self.key_features)
        return {
            group_key: {
                "keys": self.key_features,
                "threshold": self.threshold,
            },
        }

    def build(self, input_shape: dict[str, Any]) -> None:
        """Build the model based on input shape.

        Args:
            input_shape: Dictionary of input shapes for each feature.
        """
        # This method is called by Keras to build the model
        # We don't need to do anything special here since we've already
        # set up our layers in __init__, but we need to implement it
        # to avoid warnings
        super().build(input_shape)

    def call(self, inputs: dict[str, Any], training: bool = False) -> dict[str, Any]:
        """Forward pass through the model.

        Args:
            inputs: Dictionary of input tensors.
            training: Whether the model is in training mode.

        Returns:
            Dictionary of anomaly scores, flags, and reasons.
        """
        first_tensor = next(iter(inputs.values()))
        # Treat rank > 1 as batched; expand unbatched inputs.
        rank = len(getattr(first_tensor, "shape", []))
        is_batched = rank > 1 if rank else True

        processed_inputs: dict[str, Any] = {}
        if not is_batched:
            for feature, tensor in inputs.items():
                processed_inputs[feature] = keras.ops.expand_dims(tensor, 0)
            batch_size = 1
        else:
            batch_size = keras.ops.shape(first_tensor)[0]
            for feature, tensor in inputs.items():
                shape = getattr(tensor, "shape", None)
                if shape is not None and len(shape) > 1 and shape[1] == 1:
                    processed_inputs[feature] = keras.ops.reshape(tensor, [batch_size])
                else:
                    processed_inputs[feature] = tensor

        anomaly_scores = self.statistics_layer(processed_inputs, training=training)

        outputs: dict[str, Any] = {}
        for feature in self.target_features:
            feature_scores = anomaly_scores[feature]
            outputs[f"{feature}_anomaly_score"] = keras.ops.reshape(
                feature_scores["anomaly_score"],
                [batch_size, 1],
            )
            outputs[f"{feature}_is_anomaly"] = keras.ops.reshape(
                keras.ops.cast(feature_scores["is_anomaly"], "float32"),
                [batch_size, 1],
            )
            outputs[f"{feature}_reason"] = keras.ops.reshape(
                feature_scores["reason"],
                [batch_size, 1],
            )

        if not is_batched:
            for key in list(outputs):
                outputs[key] = keras.ops.squeeze(outputs[key], axis=0)
        return outputs

    def get_config(self) -> dict[str, Any]:
        """Get the model configuration.

        Returns:
            Dictionary containing the model configuration.
        """
        config = super().get_config()
        config.update(
            {
                "feature_space": self.feature_space,
                "key_features": self.key_features,
                "target_features": self.target_features,
                "min_samples": self.min_samples,
                "fallback_strategy": self.fallback_strategy,
                "threshold": self.threshold,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KerasStratifiedAnomalyModel:
        """Create a model from its configuration.

        Args:
            config: Dictionary containing the model configuration.

        Returns:
            A new instance of the model.
        """
        return cls(**config)

    def save_model(self, filepath: str) -> None:
        """Save the model to disk in Keras 3 format for TensorFlow Serving compatibility.

        This method saves the model to disk in a format that can be loaded by Keras 3
        and used with TensorFlow Serving. The model is saved with a .keras extension,
        which is required for Keras 3 compatibility.

        Args:
            filepath: Path to save the model to. If the path doesn't end with .keras,
                the extension will be added automatically.
        """
        # Ensure the filepath has a .keras extension for Keras 3 compatibility
        if not filepath.endswith(".keras"):
            filepath = filepath + ".keras"

        self.save(filepath)

    @classmethod
    def load_model(cls, filepath: str) -> KerasStratifiedAnomalyModel:
        """Load a model from disk that was saved with save_model.

        This class method loads a model that was previously saved with the save_model method.
        It ensures that the filepath has the correct extension (.keras) for Keras 3 compatibility.

        Args:
            filepath: Path to load the model from. If the path doesn't end with .keras,
                the extension will be added automatically.

        Returns:
            A loaded KerasStratifiedAnomalyModel instance with all layers and weights restored.
        """
        # Ensure the filepath has a .keras extension for Keras 3 compatibility
        if not filepath.endswith(".keras"):
            filepath = filepath + ".keras"

        return keras.models.load_model(filepath)

    def fit(self, dataset: Any, epochs: int = 1, **kwargs) -> dict[str, list[float]]:
        """Fit the model to the data.

        This method processes the dataset to compute statistics for each group.

        Args:
            dataset: TensorFlow dataset containing the training data.
            epochs: Number of epochs to train for.
            **kwargs: Additional arguments to pass to the Keras fit method.

        Returns:
            Dictionary containing training history.
        """

        # Define a custom training step that just updates statistics
        def train_step(data: dict[str, Any]) -> dict[str, Any]:
            _ = self(data, training=True)
            return {f"{feature}_loss": 0.0 for feature in self.target_features}

        # Create a simple training loop
        metrics_history: dict[str, list[float]] = {
            f"{feature}_loss": [] for feature in self.target_features
        }

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            progbar = keras.utils.Progbar(target=None)

            # Iterate over the batches of the dataset
            for step, batch in enumerate(dataset):
                # Run the training step
                metrics = train_step(batch)
                # Update the progress bar
                progbar.update(
                    step,
                    values=[
                        (k, float(v) if not hasattr(v, "numpy") else float(v.numpy()))
                        for k, v in metrics.items()
                    ],
                )

                # Update metrics history
                for k, v in metrics.items():
                    metrics_history[k].append(
                        float(v) if not hasattr(v, "numpy") else float(v.numpy()),
                    )

        return metrics_history

    def predict(self, dataset: Any) -> dict[str, dict[str, np.ndarray]]:
        """Make predictions on the data.

        Args:
            dataset: TensorFlow dataset containing the test data.

        Returns:
            Dictionary containing predictions for each sample.
        """
        # Get raw predictions
        raw_predictions = super().predict(dataset)

        # Format predictions
        predictions = {}
        for feature in self.target_features:
            predictions[feature] = {
                "anomaly_score": raw_predictions[f"{feature}_anomaly_score"],
                "is_anomaly": raw_predictions[f"{feature}_is_anomaly"] > 0.5,
                "reason": [
                    r.decode("utf-8") if isinstance(r, bytes) else str(r)
                    for r in raw_predictions[f"{feature}_reason"]
                ],
            }

        return predictions

    def generate_report(
        self,
        predictions: dict[str, dict[str, np.ndarray]],
        report_dir: str,
    ) -> str:
        """Generate a report from the predictions.

        Args:
            predictions: Dictionary containing predictions.
            report_dir: Directory to save the report to.

        Returns:
            Path to the generated report.
        """
        # Create report directory if it doesn't exist
        report_path_dir = Path(report_dir)
        report_path_dir.mkdir(parents=True, exist_ok=True)

        # Format report data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_data: dict[str, Any] = {
            "model_type": "KerasStratifiedAnomalyModel",
            "key_features": self.key_features,
            "target_features": self.target_features,
            "timestamp": timestamp,
            "predictions": {},
            "anomaly_summary": {},
            "feature_anomalies": {},
        }

        # Convert numpy arrays to lists for JSON serialization and calculate summary statistics
        for feature, feature_data in predictions.items():
            # Convert data for JSON serialization
            report_data["predictions"][feature] = {
                "anomaly_score": feature_data["anomaly_score"].tolist(),
                "is_anomaly": feature_data["is_anomaly"].tolist(),
                "reason": feature_data["reason"],
            }

            # Calculate summary statistics
            anomaly_count = np.sum(feature_data["is_anomaly"])
            total_count = len(feature_data["is_anomaly"])
            anomaly_percentage = (
                (anomaly_count / total_count) * 100 if total_count > 0 else 0
            )

            # Add to anomaly summary
            report_data["anomaly_summary"][feature] = {
                "total_samples": int(total_count),
                "anomaly_count": int(anomaly_count),
                "anomaly_percentage": float(anomaly_percentage),
                "max_anomaly_score": float(np.max(feature_data["anomaly_score"]))
                if total_count > 0
                else 0,
                "min_anomaly_score": float(np.min(feature_data["anomaly_score"]))
                if total_count > 0
                else 0,
                "mean_anomaly_score": float(np.mean(feature_data["anomaly_score"]))
                if total_count > 0
                else 0,
            }

            # Extract anomaly indices and details for feature_anomalies
            anomaly_indices = np.where(feature_data["is_anomaly"])[0]
            feature_anomalies = []

            for idx in anomaly_indices:
                # Extract the scalar value properly to avoid deprecation warning
                score_value = feature_data["anomaly_score"][idx]
                if hasattr(score_value, "item"):
                    score_value = score_value.item()

                feature_anomalies.append(
                    {
                        "index": int(idx),
                        "anomaly_score": float(score_value),
                        "reason": feature_data["reason"][idx],
                    },
                )

            # Add to feature_anomalies
            report_data["feature_anomalies"][feature] = feature_anomalies

        # Save report to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_path_dir / f"anomaly_report_{timestamp}.json"

        with report_path.open("w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Report saved to {report_path}")

        return str(report_path)
