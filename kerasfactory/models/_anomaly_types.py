"""Shared typing aliases for anomaly detection models."""

from __future__ import annotations

from typing import Any

# Feature / rule typing (matches StatsAnomaly and BusinessRulesLayer conventions)
NumericRule = tuple[str, float]
CategoricalRule = tuple[str, list[str] | str]
Rule = NumericRule | CategoricalRule
FeatureType = str  # "numerical", "cat_string", or "cat_int"
FeatureSpace = dict[str, FeatureType]
BusinessRules = dict[str, list[Rule]]

Condition = dict[str, str | float | Rule]
Requirement = dict[str, list[Rule]]
RuleCondition = dict[str, Condition | Requirement | Any]
MultiFeatureRule = dict[str, dict[str, Any]]
AggregationLevel = dict[str, dict[str, Any]]
Tensor = Any
