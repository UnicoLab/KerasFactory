"""Multi-feature business rules layer for cross-feature anomaly detection.

Evaluates declarative when/require rules that span multiple features.
"""

from __future__ import annotations

from typing import Any

from keras import KerasTensor, ops
from keras.saving import register_keras_serializable
from loguru import logger

from kerasfactory.layers._base_layer import BaseLayer

# Type aliases
FeatureType = str  # "numerical" or "categorical"
NumericRule = tuple[str, float]  # e.g. (">", 0)
CategoricalRule = tuple[str, str | list[str]]  # e.g. ("==", "winter")
FeatureRule = NumericRule | CategoricalRule
MultiFeatureRules = dict[str, dict[str, Any]]


def _as_rule(value: Any) -> FeatureRule:
    """Normalize a serialized or native rule to a 2-tuple.

    Keras JSON serialization turns tuples into lists; accept both.

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
class MultiFeatureBusinessRulesLayer(BaseLayer):
    """Evaluates business rules across multiple features for anomaly detection.

    Applies user-defined when/require rules to detect anomalies from feature
    combinations. Output keys align with ``BusinessRulesLayer`` style plus
    per-rule violation diagnostics.

    Args:
        rules: Dictionary of multi-feature rule definitions.
        feature_types: Mapping from feature name to ``"numerical"`` or
            ``"categorical"``.
        name: Optional layer name.

    Input shape:
        Dictionary mapping feature names to tensors of shape ``(batch_size, 1)``.

    Output shape:
        Dictionary with ``business_*`` keys, ``rule_violations``, and
        ``rule_names``.
    """

    def __init__(
        self,
        rules: MultiFeatureRules,
        feature_types: dict[str, FeatureType],
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the layer.

        Args:
            rules: Dictionary of multi-feature rule definitions.
            feature_types: Mapping of feature names to types.
            name: Optional layer name.
            **kwargs: Additional BaseLayer / Keras Layer arguments.

        Raises:
            ValueError: If rules have invalid structure or undefined features.
        """
        self._rules = rules
        self._feature_types = feature_types

        self._validate_params()

        self.rules = self._rules
        self.feature_types = self._feature_types

        super().__init__(name=name, **kwargs)
        logger.debug(
            f"MultiFeatureBusinessRulesLayer with {len(self.rules)} rules",
        )

    def _validate_params(self) -> None:
        """Validate rules and feature type configuration.

        Raises:
            ValueError: If structure, types, or feature references are invalid.
        """
        if not isinstance(self._rules, dict):
            raise ValueError("rules must be a dictionary")
        if not isinstance(self._feature_types, dict):
            raise ValueError("feature_types must be a dictionary")

        for feat, feat_type in self._feature_types.items():
            if feat_type not in ("numerical", "categorical"):
                raise ValueError(
                    f"Invalid feature_type for '{feat}': {feat_type}. "
                    "Must be 'numerical' or 'categorical'",
                )

        for rule_name, rule_def in self._rules.items():
            if "features" not in rule_def:
                raise ValueError(
                    f"Rule '{rule_name}' is missing required 'features' key",
                )
            if "conditions" not in rule_def:
                raise ValueError(
                    f"Rule '{rule_name}' is missing required 'conditions' key",
                )

            features = rule_def["features"]
            if not isinstance(features, list) or not features:
                raise ValueError(
                    f"Rule '{rule_name}' features must be a non-empty list",
                )
            for feat in features:
                if feat not in self._feature_types:
                    raise ValueError(
                        f"Rule '{rule_name}' references undefined feature '{feat}'",
                    )

            conditions = rule_def["conditions"]
            if not isinstance(conditions, list) or not conditions:
                raise ValueError(
                    f"Rule '{rule_name}' conditions must be a non-empty list",
                )

            for i, condition in enumerate(conditions):
                if "when" not in condition or "require" not in condition:
                    raise ValueError(
                        f"Rule '{rule_name}' condition {i} must have 'when' and "
                        "'require' keys",
                    )
                when = condition["when"]
                require = condition["require"]
                if not isinstance(when, dict) or not isinstance(require, dict):
                    raise ValueError(
                        f"Rule '{rule_name}' condition {i} when/require must be dicts",
                    )
                if not when or not require:
                    raise ValueError(
                        f"Rule '{rule_name}' condition {i} when/require "
                        "cannot be empty",
                    )
                for feat in list(when) + list(require):
                    if feat not in self._feature_types:
                        raise ValueError(
                            f"Rule '{rule_name}' condition {i} references "
                            f"undefined feature '{feat}'",
                        )
                for feat, feat_rules in require.items():
                    if not isinstance(feat_rules, list):
                        raise ValueError(
                            f"Rule '{rule_name}' condition {i} rules for "
                            f"feature '{feat}' must be a list",
                        )

    def build(self, input_shape: dict[str, tuple[int | None, ...]]) -> None:
        """Build the layer.

        Args:
            input_shape: Dictionary mapping feature names to shapes.
        """
        logger.debug(
            f"Building MultiFeatureBusinessRulesLayer with "
            f"{len(self.rules)} rules and {len(self.feature_types)} features",
        )
        super().build(input_shape)

    def compute_output_shape(
        self,
        input_shapes: dict[str, tuple[int | None, ...]],
    ) -> dict[str, tuple[int | None, ...]]:
        """Compute output shapes.

        Args:
            input_shapes: Dictionary mapping feature names to input shapes.

        Returns:
            Dictionary mapping output names to shapes.
        """
        first_feature = next(iter(input_shapes.values()))
        batch_size = first_feature[0]
        n_rules = len(self.rules)
        return {
            "business_score": (batch_size, 1),
            "business_proba": (batch_size, 1),
            "business_threshold": (),
            "business_anomaly": (batch_size, 1),
            "business_reason": (batch_size, 1),
            "rule_violations": (batch_size, n_rules),
            "rule_names": (n_rules,),
        }

    def _is_member(self, value: KerasTensor, allowed_values: list[str]) -> KerasTensor:
        """Return membership mask for categorical values.

        Args:
            value: String tensor of shape ``(batch_size, 1)``.
            allowed_values: Allowed category strings.

        Returns:
            Boolean tensor indicating membership.
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
            value: Numerical values tensor.
            rule: ``(operator, threshold)`` tuple.

        Returns:
            Boolean tensor where True means the rule is satisfied.

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
            value: Categorical values tensor.
            rule: ``(operator, allowed_values)`` tuple.

        Returns:
            Boolean tensor where True means the rule is satisfied.

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
            value: Feature values tensor.
            rules: List of feature rules.
            feature_type: ``"numerical"`` or ``"categorical"``.

        Returns:
            Boolean tensor indicating all rules are satisfied.
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

    def _ensure_batch_col(self, tensor: KerasTensor) -> KerasTensor:
        """Ensure a tensor has shape ``(batch_size, 1)``.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor with a trailing column dimension if needed.
        """
        if ops.ndim(tensor) == 1:
            return ops.expand_dims(tensor, -1)
        return tensor

    def _when_rules_for(self, value: Any, feat_type: str) -> list[FeatureRule]:
        """Build when-clause rules from a bare value or rule pair.

        Args:
            value: Bare scalar/string or ``(op, value)`` / ``[op, value]``.
            feat_type: Feature type.

        Returns:
            List with a single normalized rule.
        """
        if (
            isinstance(value, tuple | list)
            and len(value) == 2
            and isinstance(
                value[0],
                str,
            )
        ):
            return [_as_rule(value)]
        if feat_type == "categorical":
            return [("==", value)]
        return [("==", float(value))]

    def call(
        self,
        inputs: dict[str, KerasTensor],
        training: bool | None = None,
    ) -> dict[str, KerasTensor]:
        """Apply multi-feature business rules.

        Args:
            inputs: Dictionary mapping feature names to tensors.
            training: Unused; present for Keras call compatibility.

        Returns:
            Dictionary of anomaly detection outputs.

        Raises:
            ValueError: If a configured feature is missing from inputs.
        """
        del training  # Unused

        for feat in self.feature_types:
            if feat not in inputs:
                raise ValueError(f"Required feature '{feat}' not found in inputs")

        all_violations: list[KerasTensor] = []
        rule_names: list[str] = []
        rule_descriptions: list[str] = []

        for rule_name, rule_def in self.rules.items():
            rule_violations: list[KerasTensor] = []

            for condition in rule_def["conditions"]:
                when_clauses: list[KerasTensor] = []
                for feat, value in condition["when"].items():
                    feat_value = inputs[feat]
                    feat_type = self.feature_types[feat]
                    when_rule = self._when_rules_for(value, feat_type)
                    when_clauses.append(
                        self._evaluate_feature_rules(feat_value, when_rule, feat_type),
                    )

                condition_applies = when_clauses[0]
                for clause in when_clauses[1:]:
                    condition_applies = ops.logical_and(condition_applies, clause)

                requirement_compliances: list[KerasTensor] = []
                for feat, feat_rules in condition["require"].items():
                    feat_value = inputs[feat]
                    feat_type = self.feature_types[feat]
                    requirement_compliances.append(
                        self._evaluate_feature_rules(feat_value, feat_rules, feat_type),
                    )

                requirements_met = requirement_compliances[0]
                for compliance in requirement_compliances[1:]:
                    requirements_met = ops.logical_and(requirements_met, compliance)

                violation = ops.logical_and(
                    condition_applies,
                    ops.logical_not(requirements_met),
                )
                rule_violations.append(violation)

            rule_violation = rule_violations[0]
            for violation in rule_violations[1:]:
                rule_violation = ops.logical_or(rule_violation, violation)

            rule_violation = self._ensure_batch_col(rule_violation)
            all_violations.append(rule_violation)
            rule_names.append(rule_name)
            rule_descriptions.append(f"Rule '{rule_name}' violation")

        if not all_violations:
            first_feature = next(iter(inputs.values()))
            batch_size = ops.shape(first_feature)[0]
            return {
                "business_score": ops.zeros((batch_size, 1), dtype="float32"),
                "business_proba": ops.zeros((batch_size, 1), dtype="float32"),
                "business_threshold": ops.convert_to_tensor("No rules", dtype="string"),
                "business_anomaly": ops.zeros((batch_size, 1), dtype="bool"),
                "business_reason": ops.convert_to_tensor(
                    "No multi-feature business rules defined",
                    dtype="string",
                ),
                "rule_violations": ops.zeros((batch_size, 0), dtype="bool"),
                "rule_names": ops.convert_to_tensor([], dtype="string"),
            }

        # (batch, n_rules) — concatenate columns, not stack (avoids extra dim)
        stacked_violations = ops.concatenate(all_violations, axis=1)

        anomaly = all_violations[0]
        for violation in all_violations[1:]:
            anomaly = ops.logical_or(anomaly, violation)

        reason = ops.convert_to_tensor(
            "No multi-feature rule violations",
            dtype="string",
        )
        for i, violation in enumerate(all_violations):
            description = ops.convert_to_tensor(rule_descriptions[i], dtype="string")
            reason = ops.where(violation, description, reason)

        business_score = ops.cast(anomaly, "float32") * 100.0

        return {
            "business_score": business_score,
            "business_proba": business_score,
            "business_threshold": ops.convert_to_tensor(
                "Rules defined",
                dtype="string",
            ),
            "business_anomaly": anomaly,
            "business_reason": reason,
            "rule_violations": stacked_violations,
            "rule_names": ops.convert_to_tensor(rule_names, dtype="string"),
        }

    def get_config(self) -> dict[str, Any]:
        """Return the layer configuration.

        Returns:
            Dictionary containing constructor parameters.
        """
        config = super().get_config()
        config.update(
            {
                "rules": self.rules,
                "feature_types": self.feature_types,
            },
        )
        return config
