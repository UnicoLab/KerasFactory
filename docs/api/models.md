# 🤖 Models API Reference

Welcome to the KerasFactory Models documentation! All models are designed to work exclusively with **Keras 3** and provide specialized implementations for advanced machine learning tasks including time series forecasting, tabular data processing, and multimodal learning.

!!! tip "What You'll Find Here"
    Each model includes detailed documentation with:
    - ✨ **Complete parameter descriptions** with types and defaults
    - 🎯 **Usage examples** showing real-world applications
    - ⚡ **Best practices** and performance considerations
    - 🎨 **When to use** guidance for each model
    - 🔧 **Implementation notes** for developers

!!! success "Production-Ready"
    All models are fully tested, documented, and ready for production use.

!!! note "Keras 3 Compatible"
    All models are built on top of Keras base classes and are fully compatible with Keras 3.

## ⏱️ Time Series Forecasting

### 🎛️ TimeMixer
TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. 

A state-of-the-art time series forecasting model that uses decomposable components and multi-scale mixing to capture both seasonal and trend patterns at different temporal scales.

::: kerasfactory.models.TimeMixer

**Key Features:**
- Trend-seasonal decomposition (moving average or DFT)
- Multi-scale seasonal and trend mixing
- Channel-independent or dependent processing
- Support for temporal features (month, day, hour, etc.)
- Reversible instance normalization for improved training
- Multivariate time series forecasting

**Architecture:**
- Decomposition layer extracts seasonal and trend components
- Multi-scale mixing layers hierarchically combine patterns
- Encoder blocks with past decomposable mixing
- Projection layers for forecast horizon
- Reversible normalization for stable training

**References:**
- Wang, S., et al. (2023). "TimeMixer: Decomposable Multiscale Mixing For Time Series Forecasting"

### 🔀 TSMixer
TSMixer: All-MLP Architecture for Multivariate Time Series Forecasting.

An efficient all-MLP model that jointly learns temporal and cross-sectional representations through alternating temporal and feature mixing layers without attention mechanisms.

::: kerasfactory.models.TSMixer

**Key Features:**
- Temporal and feature mixing for dual-perspective learning
- Optional reversible instance normalization for training stability
- Configurable stacking of mixing layers (n_blocks parameter)
- Linear time complexity O(B × T × D²) vs attention O(B × T²)
- Multivariate time series forecasting support
- No attention mechanisms - simple, efficient, interpretable

**Architecture:**
- Instance normalization (optional reversible normalization)
- Stacked mixing layers (temporal + feature mixing per block)
- Output projection layer mapping seq_len → pred_len
- Reverse instance denormalization (optional)

**When to Use:**
- Large batch sizes or long sequences where efficiency matters
- Interpretability is important (no attention black box)
- Limited GPU memory - MLP-based is more memory efficient
- Multi-scale temporal and feature interactions needed
- Long-term forecasting with multiple related time series

**References:**
- Chen, Si-An, et al. (2023). "TSMixer: An All-MLP Architecture for Time Series Forecasting." arXiv:2303.06053

## 🏗️ Core Models

### 🚀 BaseFeedForwardModel
Flexible feed-forward model architecture for tabular data with customizable layers.

::: kerasfactory.models.feed_forward.BaseFeedForwardModel

## 🎯 Advanced Models

### 🧩 SFNEBlock
Sparse Feature Network Ensemble block for advanced feature processing and ensemble learning.

::: kerasfactory.models.SFNEBlock

### 🎭 TerminatorModel
Comprehensive tabular model that combines multiple SFNE blocks for complex data tasks.

::: kerasfactory.models.TerminatorModel

### 🔍 Autoencoder
Advanced autoencoder model for anomaly detection with optional preprocessing integration and automatic threshold configuration.

::: kerasfactory.models.autoencoder.Autoencoder

## 🔧 Base Classes

### 🏛️ BaseModel
Base class for all KerasFactory models, providing common functionality and Keras 3 compatibility.

::: kerasfactory.models._base.BaseModel

## 🚨 Anomaly Detection

### FeatureSpaceAnomalyDetectionModel
Multi-feature statistical + business-rule anomaly detection model.

::: kerasfactory.models.FeatureSpaceAnomalyDetectionModel

### KerasStratifiedAnomalyModel
End-to-end Keras stratified anomaly model using `GroupStatisticsLayer`.

::: kerasfactory.models.KerasStratifiedAnomalyModel

### StratifiedAnomalyDetectionModel
Per-group `FeatureSpaceAnomalyDetectionModel` ensemble with fallback strategies.

::: kerasfactory.models.StratifiedAnomalyDetectionModel

