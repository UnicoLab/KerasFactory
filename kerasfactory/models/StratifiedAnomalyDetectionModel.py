"""Stratified anomaly detection model (per-group FeatureSpace models).

Ported from StatsAnomaly ``stratified_model.py``.
"""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable
from loguru import logger

from kerasfactory.models._anomaly_types import FeatureSpace
from kerasfactory.models.FeatureSpaceAnomalyDetectionModel import (
    FeatureSpaceAnomalyDetectionModel,
)


@register_keras_serializable(package="kerasfactory.models")
class StratifiedAnomalyDetectionModel(keras.Model):
    """Anomaly detection model that uses stratified aggregation (group-by approach).

    This model creates separate statistical models for each unique combination of
    specified key features. For example, if 'category', 'color', and 'month' are
    specified as keys, the model will generate separate statistics for all other
    features for each unique combination of these key values.

    Attributes:
        feature_space: The feature space definition.
        stratified_config: Configuration for stratified aggregation.
        numerical_method: Method for numerical anomaly detection.
        numerical_threshold: Threshold for numerical anomaly detection.
        business_rules: Optional business rules for features.
        multi_feature_rules: Optional multi-feature business rules.
        group_statistics: Dictionary storing statistics for each group.
        group_models: Dictionary storing models for each group.
    """

    def __init__(
        self,
        feature_space: FeatureSpace,
        stratified_config: dict[str, Any],
        numerical_method: str = "z-score",
        numerical_threshold: float = 2.0,
        business_rules: dict[str, list[tuple[str, Any]]] | None = None,
        multi_feature_rules: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the stratified anomaly detection model.

        Args:
            feature_space: Dictionary mapping feature names to their types.
            stratified_config: Configuration for stratified aggregation with keys:
                - keys: List of feature names to use as grouping keys.
                - targets: List of feature names to analyze within each group.
                - min_samples_per_group: Minimum samples required for a valid group.
                - fallback_strategy: Strategy to use when a group has insufficient samples.
            numerical_method: Method for numerical anomaly detection.
                One of: "z-score", "iqr", "mad".
            numerical_threshold: Threshold for numerical anomaly detection.
            business_rules: Optional dictionary mapping feature names to their business rules.
            multi_feature_rules: Optional dictionary of multi-feature business rules.
            **kwargs: Additional keyword arguments passed to the keras.Model constructor.

        Raises:
            ValueError: If stratified_config is invalid or if key features are not in feature_space.
        """
        super().__init__(**kwargs)

        # Validate and store configuration
        self._validate_stratified_config(stratified_config, feature_space)
        self.feature_space = feature_space
        self.stratified_config = stratified_config
        self.numerical_method = numerical_method
        self.numerical_threshold = numerical_threshold
        self.business_rules = business_rules or {}
        self.multi_feature_rules = multi_feature_rules or {}

        # Initialize storage for group statistics and models
        self.group_statistics: dict[str, dict[str, Any]] = {}
        self.group_models: dict[str, FeatureSpaceAnomalyDetectionModel] = {}

        # Build the model
        self._build_model()

    def _validate_stratified_config(
        self,
        config: dict[str, Any],
        feature_space: FeatureSpace,
    ) -> None:
        """Validate the stratified configuration.

        Args:
            config: The stratified configuration to validate.
            feature_space: The feature space definition.

        Raises:
            ValueError: If the configuration is invalid.
        """
        required_keys = [
            "keys",
            "targets",
            "min_samples_per_group",
            "fallback_strategy",
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Stratified config missing required key: {key}")

        # Validate that key features exist in feature space
        for feature in config["keys"]:
            if feature not in feature_space:
                raise ValueError(f"Key feature '{feature}' not found in feature space")

        # Validate that target features exist in feature space
        for feature in config["targets"]:
            if feature not in feature_space:
                raise ValueError(
                    f"Target feature '{feature}' not found in feature space",
                )

        # Validate fallback strategy
        valid_fallback_strategies = ["global", "nearest", "default"]
        if config["fallback_strategy"] not in valid_fallback_strategies:
            raise ValueError(
                f"Invalid fallback strategy: {config['fallback_strategy']}. "
                f"Must be one of: {', '.join(valid_fallback_strategies)}",
            )

    def _build_model(self) -> None:
        """Build the model architecture."""
        # This is a placeholder - the actual models will be created during fit
        # based on the unique combinations of key features in the training data
        pass

    def _create_group_key(self, group_values: dict[str, Any]) -> str:
        """Create a string key for a group based on its values.

        Args:
            group_values: Dictionary mapping key feature names to their values.

        Returns:
            A string representation of the group key.
        """
        key_parts = []
        for feature in self.stratified_config["keys"]:
            value = group_values.get(feature, "None")
            # Convert to string and escape any special characters
            value_str = str(value).replace(":", "_").replace("|", "_")
            key_parts.append(f"{feature}:{value_str}")

        return "|".join(key_parts)

    def _extract_group_values(self, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Extract unique combinations of key features from the data.

        Args:
            data: Dictionary mapping feature names to their values.

        Returns:
            Dictionary mapping group keys to their values.
        """
        # Extract key features
        key_features = self.stratified_config["keys"]
        key_tensors = {feature: data[feature] for feature in key_features}

        # Convert tensors to numpy for easier processing
        key_arrays = {
            feature: tensor.numpy() for feature, tensor in key_tensors.items()
        }

        # Find unique combinations
        num_samples = len(key_arrays[key_features[0]])
        group_dict: dict[str, dict[str, Any]] = {}

        for i in range(num_samples):
            group_values = {
                feature: key_arrays[feature][i][0] for feature in key_features
            }
            group_key = self._create_group_key(group_values)

            if group_key not in group_dict:
                group_dict[group_key] = group_values

        return group_dict

    def _filter_data_by_group(
        self,
        data: dict[str, Any],
        group_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Filter data to include only samples matching the group values.

        Args:
            data: Dictionary mapping feature names to their values.
            group_values: Dictionary mapping key feature names to their values.

        Returns:
            Filtered data containing only samples matching the group values.
        """
        # Convert tensors to numpy for easier processing
        key_features = self.stratified_config["keys"]
        key_arrays = {feature: data[feature].numpy() for feature in key_features}

        # Create mask for matching samples
        num_samples = len(key_arrays[key_features[0]])
        mask = np.ones(num_samples, dtype=bool)

        for feature, value in group_values.items():
            feature_array = key_arrays[feature].reshape(-1)
            feature_mask = feature_array == value
            mask = mask & feature_mask

        # Apply mask to all features
        filtered_data = {}
        for feature, tensor in data.items():
            filtered_data[feature] = tf.boolean_mask(tensor, mask)

        return filtered_data

    def _create_target_feature_space(self) -> FeatureSpace:
        """Create a feature space containing only the target features.

        Returns:
            A feature space containing only the target features.
        """
        target_features = self.stratified_config["targets"]
        target_space = {
            feature: self.feature_space[feature] for feature in target_features
        }
        return target_space

    def _create_group_model(self) -> FeatureSpaceAnomalyDetectionModel:
        """Create a model for a group.

        Returns:
            A FeatureSpaceAnomalyDetectionModel for the group.
        """
        target_space = self._create_target_feature_space()

        # Filter business rules to include only target features
        target_business_rules = {}
        if self.business_rules:
            target_business_rules = {
                feature: rules
                for feature, rules in self.business_rules.items()
                if feature in target_space
            }

        # Create the model
        model = FeatureSpaceAnomalyDetectionModel(
            feature_space=target_space,
            numerical_method=self.numerical_method,
            numerical_threshold=self.numerical_threshold,
            business_rules=target_business_rules,
        )

        return model

    def fit(self, dataset: tf.data.Dataset) -> StratifiedAnomalyDetectionModel:
        """Fit the model to the training data.

        This method:
        1. Extracts all unique combinations of key features from the data
        2. For each combination, filters the data to include only matching samples
        3. Creates and fits a separate model for each group with sufficient samples

        Args:
            dataset: TensorFlow dataset containing the training data.

        Returns:
            Self for method chaining.
        """
        # Convert dataset to a dictionary of tensors for easier processing
        data: dict[str, Any] = {}
        for batch in dataset:
            for feature, tensor in batch.items():
                if feature in data:
                    data[feature] = tf.concat([data[feature], tensor], axis=0)
                else:
                    data[feature] = tensor

        # Extract unique combinations of key features
        group_dict = self._extract_group_values(data)
        logger.info(f"Found {len(group_dict)} unique groups in the data")

        # Process each group
        min_samples = self.stratified_config["min_samples_per_group"]

        for group_key, group_values in group_dict.items():
            # Filter data for this group
            group_data = self._filter_data_by_group(data, group_values)
            target_features = self.stratified_config["targets"]
            group_data = {f: group_data[f] for f in target_features if f in group_data}

            # Check if group has enough samples
            num_samples = len(next(iter(group_data.values())))
            if num_samples < min_samples:
                logger.warning(
                    f"Group {group_key} has only {num_samples} samples, "
                    f"which is less than the minimum required ({min_samples}). "
                    f"Using fallback strategy: {self.stratified_config['fallback_strategy']}",
                )
                continue

            # Create a dataset for this group
            group_dataset = tf.data.Dataset.from_tensor_slices(group_data).batch(32)

            # Create and fit a model for this group
            logger.info(
                f"Fitting model for group: {group_key} with {num_samples} samples",
            )
            group_model = self._create_group_model()
            group_model.fit(group_dataset)

            # Store the model
            self.group_models[group_key] = group_model

        # Create a global model as fallback if needed
        if self.stratified_config["fallback_strategy"] == "global":
            logger.info("Fitting global model as fallback")
            target_features = self.stratified_config["targets"]
            target_data = {f: data[f] for f in target_features if f in data}
            global_dataset = tf.data.Dataset.from_tensor_slices(target_data).batch(32)
            global_model = self._create_group_model()
            global_model.fit(global_dataset)
            self.group_models["global"] = global_model

        return self

    def predict(self, dataset: tf.data.Dataset) -> dict[str, dict[str, Any]]:
        """Make predictions on the test data.

        For each sample:
        1. Determine which group it belongs to based on key features
        2. Use the corresponding group model to make predictions
        3. If no matching group model exists, use the fallback strategy

        Args:
            dataset: TensorFlow dataset containing the test data.

        Returns:
            Dictionary of predictions organized by group and feature.
        """
        # Convert dataset to a dictionary of tensors for easier processing
        data: dict[str, Any] = {}
        for batch in dataset:
            for feature, tensor in batch.items():
                if feature in data:
                    data[feature] = tf.concat([data[feature], tensor], axis=0)
                else:
                    data[feature] = tensor

        # Extract unique combinations of key features in test data
        test_groups = self._extract_group_values(data)

        # Initialize results dictionary
        results: dict[str, dict[str, Any]] = {}

        # Process each group in test data
        for group_key, group_values in test_groups.items():
            # Filter data for this group (targets only — group models omit key features)
            group_data = self._filter_data_by_group(data, group_values)
            target_features = self.stratified_config["targets"]
            group_data = {f: group_data[f] for f in target_features if f in group_data}

            # Create a dataset for this group
            group_dataset = tf.data.Dataset.from_tensor_slices(group_data).batch(32)

            # Check if we have a model for this group
            if group_key in self.group_models:
                group_model = self.group_models[group_key]
                group_predictions = group_model.predict(group_dataset)

            elif (
                self.stratified_config["fallback_strategy"] == "global"
                and "global" in self.group_models
            ):
                logger.warning(
                    f"No model found for group {group_key}, using global model",
                )
                group_model = self.group_models["global"]
                group_predictions = group_model.predict(group_dataset)

            else:
                logger.error(
                    f"No model found for group {group_key} and no fallback available",
                )
                continue

            # Store the results
            results[group_key] = group_predictions

        return results

    def get_group_predictions(
        self,
        predictions: dict[str, dict[str, Any]],
        group_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Get predictions for a specific group.

        Args:
            predictions: Dictionary of predictions from the predict method.
            group_values: Dictionary mapping key feature names to their values.

        Returns:
            Predictions for the specified group.
        """
        group_key = self._create_group_key(group_values)
        if group_key in predictions:
            return predictions[group_key]
        else:
            logger.warning(f"No predictions found for group {group_key}")
            return {}

    def get_config(self) -> dict[str, Any]:
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "feature_space": self.feature_space,
                "stratified_config": self.stratified_config,
                "numerical_method": self.numerical_method,
                "numerical_threshold": self.numerical_threshold,
                "business_rules": self.business_rules,
                "multi_feature_rules": self.multi_feature_rules,
            },
        )
        return config
