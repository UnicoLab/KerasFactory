"""KerasFactory - A comprehensive collection of Keras 3 models, layers, and utilities.

This package provides a curated collection of state-of-the-art neural network components
built specifically for Keras 3, including models, layers, metrics, and utilities for
machine learning and deep learning applications.

Example:
    ```python
    import keras
    from kerasfactory.models import Autoencoder, BaseFeedForwardModel
    from kerasfactory.metrics import StandardDeviation, Median
    from kerasfactory.layers import TabularAttention

    # Create and use models
    autoencoder = Autoencoder(input_dim=100, encoding_dim=32)
    feed_forward = BaseFeedForwardModel(feature_names=['feat1', 'feat2'])

    # Use custom metrics
    std_metric = StandardDeviation()
    median_metric = Median()
    ```
"""

from kerasfactory import layers, models, metrics, utils

__version__ = "0.1.1"
__all__ = ["layers", "models", "metrics", "utils"]
