"""Aggregation level layer for grouped anomaly detection.

Ported from StatsAnomaly ``fixed_aggregation_level_layer.py``. Groups features
into logical aggregation levels and applies statistics and when/require rules.
"""

from __future__ import annotations

from typing import Any

from keras import KerasTensor, ops
from keras.saving import register_keras_serializable
from loguru import logger

from kerasfactory.layers._base_layer import BaseLayer

# Type aliases
FeatureType = str  # "numerical" or "categorical"
NumericRule = tuple[str, float]
CategoricalRule = tuple[str, str | list[str]]
FeatureRule = NumericRule | CategoricalRule
AggregationLevels = dict[str, dict[str, Any]]


def _as_rule(value: Any) -> FeatureRule:
    """Normalize a serialized or native rule to a 2-tuple.

    Args:
        value: Rule-like ``(op, rhs)`` or ``[op, rhs]``.

    Returns:
        Normalized ``(op, rhs)`` tuple.

    Raises:
        ValueError: If value is not a length-2 sequence.
    """
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    raise ValueError(f"Expected rule (op, value), got {value!r}")


@register_keras_serializable(package="kerasfactory.layers")
class AggregationLevelLayer(BaseLayer):
    """Evaluate rules and statistics at aggregation levels.

    Groups features into logical levels and applies per-level statistical
    aggregation of feature scores plus optional when/require rules. Emits
    per-level and global aggregation outputs.

    Args:
        aggregation_levels: Dictionary of aggregation level definitions.
        feature_types: Mapping from feature name to ``"numerical"`` or
            ``"categorical"``.
        name: Optional layer name.

    Call signature:
        ``layer(inputs, feature_stats=None)`` — pass ``feature_stats`` as a
        keyword argument so Keras ``__call__`` does not treat it as ``training``.
    """

    def __init__(
        self,
        aggregation_levels: AggregationLevels,
        feature_types: dict[str, FeatureType],
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the layer.

        Args:
            aggregation_levels: Aggregation level definitions.
            feature_types: Mapping of feature names to types.
            name: Optional layer name.
            **kwargs: Additional BaseLayer / Keras Layer arguments.

        Raises:
            ValueError: If levels are invalid or features overlap / are undefined.
        """
        self._aggregation_levels = aggregation_levels
        self._feature_types = feature_types
        self._feature_to_level: dict[str, str] = {}
        self._rule_layers: dict[str, dict[str, Any]] = {}

        self._validate_params()

        self.aggregation_levels = self._aggregation_levels
        self.feature_types = self._feature_types
        self.feature_to_level = self._feature_to_level
        self.rule_layers = self._rule_layers

        super().__init__(name=name, **kwargs)
        logger.debug(
            f"AggregationLevelLayer with {len(self.aggregation_levels)} levels",
        )

    def _validate_params(self) -> None:
        """Validate aggregation levels and build feature-to-level mapping.

        Raises:
            ValueError: If structure or feature references are invalid.
        """
        if not isinstance(self._aggregation_levels, dict):
            raise ValueError("aggregation_levels must be a dictionary")
        if not isinstance(self._feature_types, dict):
            raise ValueError("feature_types must be a dictionary")

        for feat, feat_type in self._feature_types.items():
            if feat_type not in ("numerical", "categorical"):
                raise ValueError(
                    f"Invalid feature_type for '{feat}': {feat_type}. "
                    "Must be 'numerical' or 'categorical'",
                )

        for level_name, level_def in self._aggregation_levels.items():
            if "features" not in level_def:
                raise ValueError(
                    f"Aggregation level '{level_name}' is missing required "
                    "'features' key",
                )

            features = level_def["features"]
            if not isinstance(features, list) or not features:
                raise ValueError(
                    f"Aggregation level '{level_name}' features must be a "
                    "non-empty list",
                )

            for feat in features:
                if feat not in self._feature_types:
                    raise ValueError(
                        f"Aggregation level '{level_name}' references "
                        f"undefined feature '{feat}'",
                    )
                if feat in self._feature_to_level:
                    existing_level = self._feature_to_level[feat]
                    raise ValueError(
                        f"Feature '{feat}' is defined in multiple aggregation "
                        f"levels: '{existing_level}' and '{level_name}'",
                    )
                self._feature_to_level[feat] = level_name

            if "stats" in level_def and level_def["stats"]:
                stats = level_def["stats"]
                agg_fn = stats.get("aggregation_function", "mean")
                if agg_fn not in ("mean", "max", "min", "sum"):
                    raise ValueError(
                        f"Unsupported aggregation_function '{agg_fn}' in "
                        f"level '{level_name}'",
                    )
                if "threshold" not in stats:
                    raise ValueError(
                        f"Aggregation level '{level_name}' stats missing "
                        "'threshold'",
                    )

            if "rules" in level_def and level_def["rules"]:
                self._rule_layers[level_name] = dict(level_def["rules"])

    def build(self, input_shape: dict[str, tuple[int | None, ...]]) -> None:
        """Build the layer.

        Args:
            input_shape: Dictionary mapping feature names to shapes.
        """
        logger.debug(
            f"Building AggregationLevelLayer with "
            f"{len(self.aggregation_levels)} levels",
        )
        super().build(input_shape)

    def compute_output_shape(
        self,
        input_shape: dict[str, tuple[int | None, ...]],
    ) -> dict[str, tuple[int | None, ...]]:
        """Compute output shapes.

        Args:
            input_shape: Dictionary mapping feature names to shapes.

        Returns:
            Dictionary mapping output names to shapes.
        """
        batch_size = input_shape[next(iter(input_shape))][0]
        output_shapes: dict[str, tuple[int | None, ...]] = {}

        for level_name in self.aggregation_levels:
            output_shapes.update(
                {
                    f"{level_name}_score": (batch_size, 1),
                    f"{level_name}_proba": (batch_size, 1),
                    f"{level_name}_threshold": (),
                    f"{level_name}_anomaly": (batch_size, 1),
                    f"{level_name}_reason": (batch_size, 1),
                },
            )
            if level_name in self.rule_layers:
                output_shapes.update(
                    {
                        f"{level_name}_rule_score": (batch_size, 1),
                        f"{level_name}_rule_anomaly": (batch_size, 1),
                        f"{level_name}_rule_reason": (batch_size, 1),
                    },
                )

        output_shapes.update(
            {
                "aggregation_score": (batch_size, 1),
                "aggregation_proba": (batch_size, 1),
                "aggregation_threshold": (),
                "aggregation_anomaly": (batch_size, 1),
                "aggregation_reason": (batch_size, 1),
                "aggregation_levels": (len(self.aggregation_levels),),
            },
        )
        return output_shapes

    def _is_member(self, value: KerasTensor, allowed_values: list[str]) -> KerasTensor:
        """Return membership mask for categorical values.

        Args:
            value: String tensor.
            allowed_values: Allowed category strings.

        Returns:
            Boolean membership tensor.
        """
        values_t = ops.convert_to_tensor(allowed_values, dtype="string")
        eq = ops.equal(value, ops.expand_dims(values_t, 0))
        return ops.any(eq, axis=-1, keepdims=True)

    def _evaluate_numerical_rule(
        self,
        value: KerasTensor,
        rule: FeatureRule,
    ) -> KerasTensor:
        """Evaluate a numerical rule (returns compliance).

        Args:
            value: Numerical values.
            rule: ``(operator, threshold)`` tuple.

        Returns:
            Boolean compliance tensor.

        Raises:
            ValueError: If the operator is unsupported.
        """
        op, threshold = _as_rule(rule)
        if isinstance(threshold, list):
            raise ValueError(
                f"Numerical rule threshold must be a scalar, got {threshold!r}",
            )
        threshold_t = ops.convert_to_tensor(float(threshold), dtype="float32")
        value_f = ops.cast(value, "float32")

        if op == ">":
            return ops.greater(value_f, threshold_t)
        if op == "<":
            return ops.less(value_f, threshold_t)
        if op == ">=":
            return ops.greater_equal(value_f, threshold_t)
        if op == "<=":
            return ops.less_equal(value_f, threshold_t)
        if op == "==":
            return ops.equal(value_f, threshold_t)
        if op == "!=":
            return ops.not_equal(value_f, threshold_t)
        raise ValueError(f"Unsupported numerical operator: {op}")

    def _evaluate_categorical_rule(
        self,
        value: KerasTensor,
        rule: FeatureRule,
    ) -> KerasTensor:
        """Evaluate a categorical rule (returns compliance).

        Args:
            value: Categorical values.
            rule: ``(operator, allowed_values)`` tuple.

        Returns:
            Boolean compliance tensor.

        Raises:
            ValueError: If the operator is unsupported.
        """
        op, allowed_values = _as_rule(rule)
        value_str = ops.cast(value, "string")

        if op == "==":
            if isinstance(allowed_values, list):
                return self._is_member(value_str, [str(v) for v in allowed_values])
            return ops.equal(value_str, str(allowed_values))
        if op == "!=":
            if isinstance(allowed_values, list):
                return ops.logical_not(
                    self._is_member(value_str, [str(v) for v in allowed_values]),
                )
            return ops.not_equal(value_str, str(allowed_values))
        if op == "in":
            values_list = (
                [str(v) for v in allowed_values]
                if isinstance(allowed_values, list)
                else [str(allowed_values)]
            )
            return self._is_member(value_str, values_list)
        if op == "not in":
            values_list = (
                [str(v) for v in allowed_values]
                if isinstance(allowed_values, list)
                else [str(allowed_values)]
            )
            return ops.logical_not(self._is_member(value_str, values_list))
        raise ValueError(f"Unsupported categorical operator: {op}")

    def _evaluate_feature_rules(
        self,
        value: KerasTensor,
        rules: list[FeatureRule],
        feature_type: str,
    ) -> KerasTensor:
        """Evaluate a list of rules for one feature (AND of compliances).

        Args:
            value: Feature values.
            rules: Feature rules.
            feature_type: ``"numerical"`` or ``"categorical"``.

        Returns:
            Boolean compliance tensor.
        """
        compliances: list[KerasTensor] = []
        for rule in rules:
            if feature_type == "numerical":
                compliances.append(self._evaluate_numerical_rule(value, rule))
            else:
                compliances.append(self._evaluate_categorical_rule(value, rule))

        result = compliances[0]
        for compliance in compliances[1:]:
            result = ops.logical_and(result, compliance)
        return result

    def _batch_size_from(
        self,
        inputs: dict[str, KerasTensor],
        feature_stats: dict[str, KerasTensor],
    ) -> Any:
        """Infer batch size from inputs or feature_stats.

        Args:
            inputs: Feature value tensors.
            feature_stats: Feature score tensors.

        Returns:
            Scalar batch-size tensor.

        Raises:
            ValueError: If both dicts are empty.
        """
        if inputs:
            return ops.shape(next(iter(inputs.values())))[0]
        if feature_stats:
            return ops.shape(next(iter(feature_stats.values())))[0]
        raise ValueError("Cannot infer batch size from empty inputs/feature_stats")

    def _aggregate_statistics(
        self,
        feature_stats: dict[str, KerasTensor],
        level_name: str,
        level_def: dict[str, Any],
        batch_size: Any,
    ) -> dict[str, KerasTensor]:
        """Aggregate feature-level scores for one aggregation level.

        Args:
            feature_stats: Mapping of feature name to score tensor.
            level_name: Aggregation level name.
            level_def: Level definition.
            batch_size: Batch size scalar tensor.

        Returns:
            Per-level statistical outputs.

        Raises:
            ValueError: If aggregation_function is unsupported.
        """
        if "stats" not in level_def or not level_def["stats"]:
            return {
                f"{level_name}_score": ops.zeros((batch_size, 1), dtype="float32"),
                f"{level_name}_proba": ops.zeros((batch_size, 1), dtype="float32"),
                f"{level_name}_threshold": ops.convert_to_tensor(
                    "No stats defined",
                    dtype="string",
                ),
                f"{level_name}_anomaly": ops.zeros((batch_size, 1), dtype="bool"),
                f"{level_name}_reason": ops.convert_to_tensor(
                    f"No statistics defined for {level_name}",
                    dtype="string",
                ),
            }

        stats_config = level_def["stats"]
        features = level_def["features"]
        agg_function = stats_config["aggregation_function"]
        threshold = stats_config["threshold"]

        feature_scores = [
            feature_stats[feat] for feat in features if feat in feature_stats
        ]

        if not feature_scores:
            return {
                f"{level_name}_score": ops.zeros((batch_size, 1), dtype="float32"),
                f"{level_name}_proba": ops.zeros((batch_size, 1), dtype="float32"),
                f"{level_name}_threshold": ops.convert_to_tensor(
                    str(threshold),
                    dtype="string",
                ),
                f"{level_name}_anomaly": ops.zeros((batch_size, 1), dtype="bool"),
                f"{level_name}_reason": ops.convert_to_tensor(
                    f"No feature statistics available for {level_name}",
                    dtype="string",
                ),
            }

        stacked_scores = ops.stack(feature_scores, axis=1)

        if agg_function == "mean":
            agg_score = ops.mean(stacked_scores, axis=1, keepdims=True)
        elif agg_function == "max":
            agg_score = ops.max(stacked_scores, axis=1, keepdims=True)
        elif agg_function == "min":
            agg_score = ops.min(stacked_scores, axis=1, keepdims=True)
        elif agg_function == "sum":
            agg_score = ops.sum(stacked_scores, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_function}")

        while ops.ndim(agg_score) > 2:
            agg_score = ops.squeeze(agg_score, axis=-1)
            if ops.ndim(agg_score) == 1:
                agg_score = ops.expand_dims(agg_score, -1)

        anomaly = ops.cast(ops.greater(agg_score, threshold), dtype="bool")
        reason = ops.where(
            anomaly,
            ops.convert_to_tensor(
                f"{level_name} anomaly detected ({agg_function} of feature scores)",
                dtype="string",
            ),
            ops.convert_to_tensor(f"No {level_name} anomaly detected", dtype="string"),
        )
        proba = ops.sigmoid(agg_score - threshold) * 100.0

        return {
            f"{level_name}_score": agg_score,
            f"{level_name}_proba": proba,
            f"{level_name}_threshold": ops.convert_to_tensor(
                str(threshold),
                dtype="string",
            ),
            f"{level_name}_anomaly": anomaly,
            f"{level_name}_reason": reason,
        }

    def _evaluate_level_rules(
        self,
        inputs: dict[str, KerasTensor],
        level_name: str,
        level_def: dict[str, Any],
        batch_size: Any,
    ) -> dict[str, KerasTensor]:
        """Evaluate when/require rules for one aggregation level.

        Args:
            inputs: Feature value tensors.
            level_name: Aggregation level name.
            level_def: Level definition.
            batch_size: Batch size scalar tensor.

        Returns:
            Per-level rule outputs.

        Raises:
            ValueError: If a required rule feature is missing.
        """
        if "rules" not in level_def or not level_def["rules"]:
            return {
                f"{level_name}_rule_score": ops.zeros((batch_size, 1), dtype="float32"),
                f"{level_name}_rule_anomaly": ops.zeros((batch_size, 1), dtype="bool"),
                f"{level_name}_rule_reason": ops.convert_to_tensor(
                    "No rules defined",
                    dtype="string",
                ),
            }

        rules = level_def["rules"]
        for rule_name, rule_def in rules.items():
            for condition in rule_def.get("conditions", []):
                for feat in list(condition.get("when", {}).keys()) + list(
                    condition.get("require", {}).keys(),
                ):
                    if feat not in inputs:
                        raise ValueError(
                            f"Required feature '{feat}' for rule '{rule_name}' "
                            "not found in inputs",
                        )

        all_violations: list[KerasTensor] = []
        rule_names: list[str] = []

        for rule_name, rule_def in rules.items():
            rule_violations = ops.zeros((batch_size, 1), dtype="bool")

            for condition in rule_def.get("conditions", []):
                when_conditions = condition.get("when", {})
                when_satisfied: KerasTensor | None = None

                for feat, value in when_conditions.items():
                    feat_value = inputs[feat]
                    feat_type = self.feature_types[feat]

                    if feat_type == "numerical":
                        if (
                            isinstance(value, tuple | list)
                            and len(value) == 2
                            and (isinstance(value[0], str))
                        ):
                            feat_condition = self._evaluate_numerical_rule(
                                feat_value,
                                _as_rule(value),
                            )
                        else:
                            feat_condition = self._evaluate_numerical_rule(
                                feat_value,
                                ("==", float(value)),
                            )
                    else:
                        if (
                            isinstance(value, tuple | list)
                            and len(value) == 2
                            and (isinstance(value[0], str))
                        ):
                            feat_condition = self._evaluate_categorical_rule(
                                feat_value,
                                _as_rule(value),
                            )
                        else:
                            feat_condition = ops.equal(
                                ops.cast(feat_value, "string"),
                                str(value),
                            )

                    if when_satisfied is None:
                        when_satisfied = feat_condition
                    else:
                        when_satisfied = ops.logical_and(when_satisfied, feat_condition)

                if when_satisfied is None:
                    when_satisfied = ops.ones((batch_size, 1), dtype="bool")

                require_conditions = condition.get("require", {})
                require_satisfied: KerasTensor | None = None

                for feat, feat_rules in require_conditions.items():
                    feat_value = inputs[feat]
                    feat_type = self.feature_types[feat]
                    feat_compliance = self._evaluate_feature_rules(
                        feat_value,
                        feat_rules,
                        feat_type,
                    )
                    if require_satisfied is None:
                        require_satisfied = feat_compliance
                    else:
                        require_satisfied = ops.logical_and(
                            require_satisfied,
                            feat_compliance,
                        )

                if require_satisfied is None:
                    require_satisfied = ops.ones((batch_size, 1), dtype="bool")

                condition_violation = ops.logical_and(
                    when_satisfied,
                    ops.logical_not(require_satisfied),
                )
                rule_violations = ops.logical_or(rule_violations, condition_violation)

            all_violations.append(rule_violations)
            rule_names.append(rule_name)

        anomaly = all_violations[0]
        for violation in all_violations[1:]:
            anomaly = ops.logical_or(anomaly, violation)

        reason = ops.convert_to_tensor("No rule violations", dtype="string")
        for i, violation in enumerate(all_violations):
            description = ops.convert_to_tensor(
                f"{level_name} rule '{rule_names[i]}' violation",
                dtype="string",
            )
            reason = ops.where(violation, description, reason)

        rule_score = ops.cast(anomaly, "float32") * 100.0
        return {
            f"{level_name}_rule_score": rule_score,
            f"{level_name}_rule_anomaly": anomaly,
            f"{level_name}_rule_reason": reason,
        }

    def call(
        self,
        inputs: dict[str, KerasTensor],
        feature_stats: dict[str, KerasTensor] | None = None,
        training: bool | None = None,
    ) -> dict[str, KerasTensor]:
        """Apply aggregation-level statistics and rules.

        Args:
            inputs: Dictionary mapping feature names to value tensors.
            feature_stats: Optional mapping of feature names to statistical
                scores. Prefer keyword argument ``feature_stats=...``.
            training: Unused; present for Keras call compatibility.

        Returns:
            Dictionary of per-level and global aggregation outputs.

        Raises:
            ValueError: If a required feature for a level is missing.
        """
        del training  # Unused

        for level_name, level_def in self.aggregation_levels.items():
            for feat in level_def["features"]:
                if feat not in inputs:
                    raise ValueError(
                        f"Required feature '{feat}' for aggregation level "
                        f"'{level_name}' not found in inputs",
                    )

        if feature_stats is None:
            feature_stats = {}

        batch_size = self._batch_size_from(inputs, feature_stats)
        level_outputs: dict[str, KerasTensor] = {}
        level_anomalies: list[KerasTensor] = []
        level_reasons: list[KerasTensor] = []

        for level_name, level_def in self.aggregation_levels.items():
            stats_results = self._aggregate_statistics(
                feature_stats,
                level_name,
                level_def,
                batch_size,
            )
            level_outputs.update(stats_results)

            rule_results = self._evaluate_level_rules(
                inputs,
                level_name,
                level_def,
                batch_size,
            )
            level_outputs.update(rule_results)

            # Keep `{level}_anomaly` as statistical-only (fixed-source behavior)
            stats_anomaly = stats_results[f"{level_name}_anomaly"]
            rule_anomaly = rule_results.get(
                f"{level_name}_rule_anomaly",
                ops.zeros_like(stats_anomaly, dtype="bool"),
            )
            combined_anomaly = ops.logical_or(stats_anomaly, rule_anomaly)

            stats_reason = stats_results[f"{level_name}_reason"]
            rule_reason = rule_results.get(
                f"{level_name}_rule_reason",
                ops.convert_to_tensor("No rule violations", dtype="string"),
            )
            combined_reason = ops.where(
                rule_anomaly,
                rule_reason,
                ops.where(
                    stats_anomaly,
                    stats_reason,
                    ops.convert_to_tensor(
                        f"No anomalies in {level_name}",
                        dtype="string",
                    ),
                ),
            )
            level_outputs[f"{level_name}_reason"] = combined_reason

            level_anomalies.append(combined_anomaly)
            level_reasons.append(combined_reason)

        if level_anomalies:
            global_anomaly = level_anomalies[0]
            for anomaly in level_anomalies[1:]:
                global_anomaly = ops.logical_or(global_anomaly, anomaly)

            global_reason = ops.convert_to_tensor(
                "No aggregation level anomalies",
                dtype="string",
            )
            for anomaly, reason in zip(level_anomalies, level_reasons, strict=False):
                global_reason = ops.where(anomaly, reason, global_reason)

            global_score = ops.cast(global_anomaly, "float32") * 100.0
            level_outputs.update(
                {
                    "aggregation_score": global_score,
                    "aggregation_proba": global_score,
                    "aggregation_threshold": ops.convert_to_tensor(
                        "Aggregation rules",
                        dtype="string",
                    ),
                    "aggregation_anomaly": global_anomaly,
                    "aggregation_reason": global_reason,
                    "aggregation_levels": ops.convert_to_tensor(
                        list(self.aggregation_levels.keys()),
                        dtype="string",
                    ),
                },
            )
        else:
            level_outputs.update(
                {
                    "aggregation_score": ops.zeros((batch_size, 1), dtype="float32"),
                    "aggregation_proba": ops.zeros((batch_size, 1), dtype="float32"),
                    "aggregation_threshold": ops.convert_to_tensor(
                        "No aggregation levels",
                        dtype="string",
                    ),
                    "aggregation_anomaly": ops.zeros((batch_size, 1), dtype="bool"),
                    "aggregation_reason": ops.convert_to_tensor(
                        "No aggregation levels defined",
                        dtype="string",
                    ),
                    "aggregation_levels": ops.convert_to_tensor([], dtype="string"),
                },
            )

        return level_outputs

    def get_config(self) -> dict[str, Any]:
        """Return the layer configuration.

        Returns:
            Dictionary containing constructor parameters.
        """
        config = super().get_config()
        config.update(
            {
                "aggregation_levels": self.aggregation_levels,
                "feature_types": self.feature_types,
            },
        )
        return config
