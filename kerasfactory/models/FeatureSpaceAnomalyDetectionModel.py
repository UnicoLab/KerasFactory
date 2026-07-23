#!/usr/bin/env python
"""Feature Space Anomaly Detection Model.

This module implements a flexible anomaly detection model that can handle both numerical
and categorical features. The model combines statistical anomaly detection with
business rule validation.

The model architecture consists of three main components for each feature:
1. Statistical Anomaly Detection: Detects anomalies based on statistical properties
2. Business Rules: Validates data against user-defined business rules
3. Output Merger: Combines statistical and business rule results

The outputs are returned as a dictionary with keys:
  - "proba": Anomaly probability (0–100)
  - "score": Normalized deviation score or rule violation magnitude
  - "threshold": The threshold (or allowed range description)
  - "anomaly": Boolean flag indicating anomaly
  - "reason": A string explaining why the feature was flagged as anomalous

Typical usage example:

    feature_space = {
        "temperature": "numerical",
        "color": "cat_string"
    }
    business_rules = {
        "temperature": [(">" , 0), ("<", 100)],
        "color": [("==", ["red", "green", "blue"])]
    }
    model = FeatureSpaceAnomalyDetectionModel(
        feature_space=feature_space,
        numerical_threshold=2.0,
        business_rules=business_rules
    )
    model.fit(train_dataset)
    predictions = model.predict(test_dataset)
"""

from __future__ import annotations

from typing import Any, cast

import keras
from keras import layers, ops
from keras.saving import register_keras_serializable
from loguru import logger

from kerasfactory.layers import (
    AggregationLevelLayer,
    BusinessRulesLayer,
    CategoricalAnomalyDetectionLayer,
    DistributionTransformLayer,
    GlobalAnomalyMergeLayer,
    MultiFeatureBusinessRulesLayer,
    StatisticalNumericalAnomalyDetection,
)
from kerasfactory.layers.BusinessRulesLayer import Rule
from kerasfactory.layers.StatisticalNumericalAnomalyDetection import StatisticalMethod
from kerasfactory.models._anomaly_types import (
    AggregationLevel,
    BusinessRules,
    FeatureSpace,
    MultiFeatureRule,
    Tensor,
)

# Keras ops alias used throughout the original StatsAnomaly model
KO = ops


###############################################################################
# FeatureSpaceAnomalyDetectionModel (updated to include business rules)
###############################################################################
@register_keras_serializable(package="kerasfactory.models")
class FeatureSpaceAnomalyDetectionModel(keras.Model):
    """Multi-feature anomaly detection model with business rules support.

    This model implements a flexible architecture for detecting anomalies across multiple
    features of different types (numerical or categorical). For each feature, it combines
    statistical anomaly detection with optional business rule validation.

    The model creates a parallel processing pipeline for each feature:
    1. Statistical Branch:
       - For numerical features: Z-score based anomaly detection
       - For categorical features: Valid category membership checking
    2. Business Rules Branch (optional):
       - Validates values against user-defined rules
       - Provides pass-through defaults if no rules specified
    3. Merger:
       - Combines statistical and business rule results

    Attributes:
        feature_space: Dictionary mapping feature names to their types.
        numerical_threshold: Z-score threshold for numerical features.
        business_rules: Dictionary mapping features to their business rules.
        stat_layers: Dictionary of statistical anomaly detection layers.
        business_layers: Dictionary of business rule validation layers.
        merge_layers: Dictionary of output merger layers.

    Example:
        ```python
        # Define feature space and rules
        feature_space = {
            'temperature': 'numerical',
            'color': 'cat_string'
        }
        business_rules = {
            'temperature': [('>', 0), ('<', 100)],
            'color': [('in', ['red', 'green', 'blue'])]
        }

        # Create and train model
        model = FeatureSpaceAnomalyDetectionModel(
            feature_space=feature_space,
            numerical_threshold=2.0,
            business_rules=business_rules
        )
        model.fit(train_dataset)

        # Make predictions
        predictions = model.predict({
            'temperature': [[20.0], [150.0]],
            'color': [['red'], ['purple']]
        })
        ```
    """

    def __init__(
        self,
        feature_space: FeatureSpace,
        numerical_threshold: float = 2.0,
        business_rules: BusinessRules | None = None,
        multi_feature_rules: MultiFeatureRule | None = None,
        aggregation_levels: AggregationLevel | None = None,
        numerical_method: StatisticalMethod | str = "z-score",
        transformations: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        """Initializes the model.

        Args:
            feature_space: Dictionary mapping feature names to their types.
            numerical_threshold: Z-score threshold for numerical features.
            business_rules: Optional dictionary mapping features to their rules.
            multi_feature_rules: Optional dictionary of rules that apply to combinations of features.
                Each rule should have the following structure:
                {
                    "rule_name": {
                        "features": ["feature1", "feature2"],
                        "conditions": [
                            {
                                "when": {"feature1": "value"},
                                "require": {"feature2": [(">", 0)]}
                            }
                        ]
                    }
                }
            aggregation_levels: Optional dictionary defining aggregation levels for features.
                Each aggregation level should have the following structure:
                {
                    "level_name": {  # Name of the aggregation level (e.g. "lengths")
                        "features": ["feature1", "feature2"],  # List of features in this aggregation level
                        "rules": {  # Optional rules specific to this aggregation level
                            "rule_name": {
                                "conditions": [
                                    {
                                        "when": {"feature1": "value"},
                                        "require": {"feature2": [(">", 0)]}
                                    }
                                ]
                            }
                        },
                        "stats": {  # Optional statistics configuration for this aggregation level
                            "method": "z-score",  # Statistical method to use
                            "threshold": 2.0,  # Threshold for anomaly detection
                            "aggregation_function": "mean"  # How to aggregate feature-level statistics
                        }
                    }
                }
            numerical_method: Statistical method for numerical anomaly detection.
            transformations: Optional dictionary specifying transformations for features.
            **kwargs: Additional model arguments.

        Raises:
            ValueError: If feature_space contains invalid feature types.
        """
        self.feature_space = feature_space
        self.numerical_threshold = numerical_threshold
        self.business_rules = business_rules if business_rules is not None else {}
        self.multi_feature_rules = (
            multi_feature_rules if multi_feature_rules is not None else {}
        )
        self.aggregation_levels = (
            aggregation_levels if aggregation_levels is not None else {}
        )
        self.numerical_method = numerical_method
        self.transformations = transformations if transformations is not None else {}

        inputs: dict[str, Any] = {}
        merged_outputs: dict[str, Any] = {}
        self.stat_layers: dict[str, Any] = {}
        self.business_layers: dict[str, Any] = {}
        self.transform_layers: dict[str, Any] = {}
        self.merge_layers: dict[str, Any] = {}
        self.multi_feature_business_layer = None
        self.aggregation_level_layer = None

        for feat, ftype in feature_space.items():
            # Set input dtype based on feature type
            if ftype == "numerical":
                dtype = "float32"
            elif ftype.startswith("cat_string"):
                dtype = "string"
            elif ftype.startswith("cat_int"):
                dtype = "int32"
            else:
                raise ValueError(f"Unsupported feature type: {ftype}")

            inp = keras.Input(shape=(1,), name=feat, dtype=dtype)
            inputs[feat] = inp

            # Add transformation layer if specified
            if feat in self.transformations:
                transform_config = self.transformations[feat]
                transform_layer = DistributionTransformLayer(
                    transform_type=transform_config.get("type", "none"),
                    lambda_param=transform_config.get("lambda", 0.0),
                    name=f"{feat}_transform_layer",
                )
                self.transform_layers[feat] = transform_layer
                transformed_inp = transform_layer(inp)
            else:
                transformed_inp = inp

            # Statistical branch:
            if ftype == "numerical":
                stat_layer = StatisticalNumericalAnomalyDetection(
                    method=cast(StatisticalMethod, self.numerical_method),
                    threshold=numerical_threshold,
                    name=f"{feat}_num_layer",
                )
            elif ftype.startswith("cat"):
                stat_layer = CategoricalAnomalyDetectionLayer(
                    dtype=ftype.split("_")[1],
                    name=f"{feat}_cat_layer",
                )
            else:
                raise ValueError("Unsupported feature type: " + ftype)
            self.stat_layers[feat] = stat_layer
            stat_out = stat_layer(
                transformed_inp if feat in self.transformations else inp,
            )
            # Business branch:
            if feat in self.business_rules:
                bus_layer = BusinessRulesLayer(
                    rules=cast(list[Rule], self.business_rules[feat]),
                    feature_type="numerical" if ftype == "numerical" else "categorical",
                    name=f"{feat}_business_layer",
                )
            else:
                # Default: no business rule violation
                def default_bus(x: Tensor) -> dict[str, Tensor]:
                    """Default business rule layer that always returns no violation.

                    Args:
                        x: Input tensor of shape (batch_size, 1).

                    Returns:
                        Dictionary of default business rule outputs.
                    """
                    batch_size = ops.shape(x)[0]
                    return {
                        "business_score": ops.zeros((batch_size, 1), dtype="float32"),
                        "business_proba": ops.zeros((batch_size, 1), dtype="float32"),
                        "business_threshold": KO.convert_to_tensor(
                            "None",
                            dtype="string",
                        ),
                        "business_anomaly": ops.zeros((batch_size, 1), dtype="bool"),
                        "business_reason": KO.convert_to_tensor(
                            "No business rule",
                            dtype="string",
                        ),
                        "business_value": x,
                    }

                def default_bus_output_shape(
                    input_shape: tuple[int | None, int],
                ) -> dict[str, tuple[Any, ...]]:
                    """Computes output shapes for default business rule layer.

                    Args:
                        input_shape: Shape of input tensor (batch_size, 1).

                    Returns:
                        Dictionary mapping output names to their shapes.
                    """
                    return {
                        "business_score": (input_shape[0], 1),
                        "business_proba": (input_shape[0], 1),
                        "business_threshold": (1,),
                        "business_anomaly": (input_shape[0], 1),
                        "business_reason": (input_shape[0], 1),
                        "business_value": input_shape,
                    }

                bus_layer = layers.Lambda(
                    default_bus,
                    output_shape=default_bus_output_shape,
                    name=f"{feat}_business_layer",
                )
            self.business_layers[feat] = bus_layer
            business_out = bus_layer(inp)
            # Prepare outputs for merging
            is_anomaly = KO.logical_or(
                stat_out["anomaly"],
                business_out["business_anomaly"],
            )

            # Handle reason string separately to avoid type promotion issues
            business_reason = KO.cast(business_out["business_reason"], dtype="string")
            stat_reason = KO.cast(stat_out["reason"], dtype="string")

            # Create a string tensor for each case
            if ftype == "numerical":
                # For numerical features
                reason = KO.where(
                    is_anomaly,
                    KO.where(
                        business_out["business_anomaly"],
                        business_reason,
                        stat_reason,
                    ),
                    KO.convert_to_tensor(["No anomalies detected"], dtype="string"),
                )
            else:
                # For categorical features
                reason = KO.where(
                    business_out["business_anomaly"],
                    business_reason,
                    KO.where(
                        stat_out["anomaly"],
                        stat_reason,
                        KO.convert_to_tensor(["No anomalies detected"], dtype="string"),
                    ),
                )

            merged = {
                "probability": KO.maximum(
                    stat_out["proba"],
                    business_out["business_proba"],
                ),
                "score": KO.maximum(stat_out["score"], business_out["business_score"]),
                "is_anomaly": is_anomaly,
                "reason": KO.reshape(
                    KO.where(
                        KO.logical_and(is_anomaly, business_out["business_anomaly"]),
                        KO.convert_to_tensor(
                            ["Business rule violation"],
                            dtype="string",
                        ),
                        reason,
                    ),
                    (-1,),
                ),
                "input_value": inp,
                "details": layers.Lambda(
                    lambda x: KO.tile(
                        KO.reshape(
                            KO.convert_to_tensor(
                                ["No additional details"],
                                dtype="string",
                            ),
                            (1, 1),
                        ),
                        [KO.shape(x)[0], 1],
                    ),
                    output_shape=lambda input_shape: (input_shape[0], 1),
                )(inp),
            }
            merged_outputs[feat] = merged

        # Add multi-feature business rules layer if rules are provided
        if self.multi_feature_rules:
            # Create a multi-feature business rules layer
            feature_types = {}
            for feat, ftype in self.feature_space.items():
                if ftype == "numerical":
                    feature_types[feat] = "numerical"
                elif ftype.startswith("cat_"):
                    feature_types[feat] = "categorical"
                else:
                    raise ValueError(
                        f"Unsupported feature type for multi-feature rules: {ftype}",
                    )

            self.multi_feature_business_layer = MultiFeatureBusinessRulesLayer(
                rules=self.multi_feature_rules,
                feature_types=feature_types,
                name="multi_feature_business_layer",
            )

            # Apply the layer to all inputs
            multi_feature_outputs = self.multi_feature_business_layer(inputs)

            # Create a merged output for the multi-feature rules
            multi_feature_merged = {
                "probability": multi_feature_outputs["business_proba"],
                "score": multi_feature_outputs["business_score"],
                "is_anomaly": multi_feature_outputs["business_anomaly"],
                "reason": multi_feature_outputs["business_reason"],
                "input_value": layers.Lambda(
                    lambda x: KO.tile(
                        KO.reshape(
                            KO.convert_to_tensor(["Multiple features"], dtype="string"),
                            (1, 1),
                        ),
                        [KO.shape(x)[0], 1],
                    ),
                    output_shape=lambda input_shape: (input_shape[0], 1),
                )(next(iter(inputs.values()))),
                "details": layers.Lambda(
                    lambda x: KO.tile(
                        KO.reshape(
                            KO.convert_to_tensor(
                                ["Multi-feature rule violation"],
                                dtype="string",
                            ),
                            (1, 1),
                        ),
                        [KO.shape(x)[0], 1],
                    ),
                    output_shape=lambda input_shape: (input_shape[0], 1),
                )(next(iter(inputs.values()))),
            }

            # Add to merged outputs
            merged_outputs["multi_feature"] = multi_feature_merged

        # Add aggregation level layer if aggregation levels are provided
        if self.aggregation_levels:
            # Create feature types dictionary for aggregation level layer
            feature_types = {}
            for feat, ftype in self.feature_space.items():
                if ftype == "numerical":
                    feature_types[feat] = "numerical"
                elif ftype.startswith("cat_"):
                    feature_types[feat] = "categorical"
                else:
                    raise ValueError(
                        f"Unsupported feature type for aggregation levels: {ftype}",
                    )

            # Create the aggregation level layer
            self.aggregation_level_layer = AggregationLevelLayer(
                aggregation_levels=self.aggregation_levels,
                feature_types=feature_types,
                name="aggregation_level_layer",
            )

            # Collect feature-level statistical scores for aggregation
            feature_stats = {}
            for feat in self.stat_layers:
                feature_stats[feat] = merged_outputs[feat]["score"]

            # Apply the aggregation level layer
            aggregation_outputs = self.aggregation_level_layer(inputs, feature_stats)

            # Add aggregation level outputs to merged outputs
            for key, value in aggregation_outputs.items():
                if key.startswith("aggregation_"):
                    # Add global aggregation outputs directly
                    merged_outputs[key] = value
                else:
                    # For level-specific outputs, create a new entry in merged_outputs
                    level_name = key.split("_")[0]
                    if level_name not in merged_outputs:
                        merged_outputs[level_name] = {}

                    # Extract the output type (score, anomaly, etc.)
                    output_type = "_".join(key.split("_")[1:])
                    merged_outputs[level_name][output_type] = value

        # Add global merge layer to combine predictions across features
        global_merge = GlobalAnomalyMergeLayer(
            merge_strategy="max",
            format_details=True,
            name="global_merge",
        )
        global_output = global_merge(merged_outputs)
        merged_outputs["global"] = global_output

        super().__init__(inputs=inputs, outputs=merged_outputs, **kwargs)

    def compute_global_stats(self, dataset: Any) -> dict[str, dict[str, Any]]:
        """Computes global statistics from a training dataset.

        For numerical features, computes statistics based on the selected method.
        For categorical features, builds vocabulary of valid values.

        Args:
            dataset: A tf.data.Dataset containing feature tensors.

        Returns:
            A dictionary mapping feature names to their statistics.
        """
        global_stats: dict[str, dict[str, Any]] = {}
        for feat, ftype in self.feature_space.items():
            if ftype == "numerical":
                # Initialize with basic statistics needed for all methods
                global_stats[feat] = {
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "count": 0,
                    "values": [],  # Store values for percentile calculations
                }
            elif ftype.startswith("cat"):
                global_stats[feat] = {"vocab": set()}

        for batch in dataset:
            for feat, ftype in self.feature_space.items():
                values = batch[feat]
                if ftype == "numerical":
                    values = KO.convert_to_tensor(values, dtype="float32")

                    # Apply transformation if specified
                    if feat in self.transformations and feat in self.transform_layers:
                        values = self.transform_layers[feat](values)

                    values = ops.reshape(values, [-1])
                    batch_sum = float(KO.sum(values).numpy().item())
                    batch_sum_sq = float(KO.sum(KO.square(values)).numpy().item())
                    batch_count = int(ops.size(values).numpy().item())

                    global_stats[feat]["sum"] += batch_sum
                    global_stats[feat]["sum_sq"] += batch_sum_sq
                    global_stats[feat]["count"] += batch_count

                    # Store values for percentile calculations if needed
                    if self.numerical_method in ["iqr", "mad"]:
                        global_stats[feat]["values"].extend(values.numpy().tolist())

                elif ftype.startswith("cat"):
                    arr = values.numpy().tolist()
                    if isinstance(arr, list) and arr and isinstance(arr[0], list):
                        arr = [item for sublist in arr for item in sublist]
                    global_stats[feat]["vocab"].update(arr)

        for feat, ftype in self.feature_space.items():
            if ftype == "numerical":
                tot = global_stats[feat]["count"]
                if tot == 0:
                    raise ValueError(f"No data for numerical feature {feat}")

                # Calculate basic statistics
                mean = global_stats[feat]["sum"] / tot
                variance = global_stats[feat]["sum_sq"] / tot - mean**2
                std = (variance**0.5) if variance > 0 else 1e-6
                global_stats[feat]["mean"] = mean
                global_stats[feat]["std"] = std

                # Calculate additional statistics for IQR and MAD methods
                if self.numerical_method in ["iqr", "mad"]:
                    values = sorted(global_stats[feat]["values"])
                    n = len(values)

                    # Calculate median
                    if n % 2 == 0:
                        median = (values[n // 2 - 1] + values[n // 2]) / 2
                    else:
                        median = values[n // 2]
                    global_stats[feat]["median"] = median

                    # Calculate quartiles for IQR
                    if self.numerical_method == "iqr":
                        q1_idx = n // 4
                        q3_idx = 3 * n // 4
                        global_stats[feat]["q1"] = values[q1_idx]
                        global_stats[feat]["q3"] = values[q3_idx]

                    # Calculate MAD
                    if self.numerical_method == "mad":
                        deviations = [abs(x - median) for x in values]
                        deviations.sort()
                        if n % 2 == 0:
                            mad = (deviations[n // 2 - 1] + deviations[n // 2]) / 2
                        else:
                            mad = deviations[n // 2]
                        global_stats[feat]["mad"] = max(mad, 1e-6)  # Ensure non-zero

                # Remove the values list to save memory
                if "values" in global_stats[feat]:
                    del global_stats[feat]["values"]

            elif ftype.startswith("cat"):
                global_stats[feat]["vocab"] = sorted(global_stats[feat]["vocab"])

        return global_stats

    def fit(self, dataset: Any, *args: Any, **kwargs: Any) -> Any:
        """Fits the model to the dataset.

        This method first computes global statistics from the dataset, then
        initializes all layers with these statistics.

        Args:
            dataset: A tf.data.Dataset containing feature tensors.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A dummy history object for compatibility with Keras.
        """
        global_stats = self.compute_global_stats(dataset)
        for feat, ftype in self.feature_space.items():
            stat_layer = self.stat_layers[feat]
            if ftype == "numerical":
                stat_layer.initialize_from_stats(global_stats[feat])
            elif ftype.startswith("cat"):
                stat_layer.initialize_from_stats(global_stats[feat]["vocab"])
        logger.info("Model adaptation complete. Ready for training/inference.")
        return self

    def get_config(self) -> dict[str, Any]:
        """Return the model configuration for serialization.

        Returns:
            Dictionary of constructor arguments suitable for ``from_config``.
        """
        config = super().get_config()
        config.update(
            {
                "feature_space": self.feature_space,
                "numerical_threshold": self.numerical_threshold,
                "business_rules": self.business_rules,
                "multi_feature_rules": self.multi_feature_rules,
                "aggregation_levels": self.aggregation_levels,
                "numerical_method": self.numerical_method,
                "transformations": self.transformations,
            },
        )
        return config
