"""Data generation utilities for KerasFactory model testing and demonstrations."""

from typing import TYPE_CHECKING, Union
import numpy as np
import keras

# Optional TensorFlow import for tf.data.Dataset functionality
try:
    import tensorflow as tf

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    tf = None

# Type checking imports
if TYPE_CHECKING:
    if _TENSORFLOW_AVAILABLE:
        from tensorflow.data import Dataset
    else:
        Dataset = None  # type: ignore


class KerasFactoryDataGenerator:
    """Utility class for generating synthetic datasets for KerasFactory model testing."""

    @staticmethod
    def generate_regression_data(
        n_samples: int = 1000,
        n_features: int = 10,
        noise_level: float = 0.1,
        random_state: int = 42,
        include_interactions: bool = True,
        include_nonlinear: bool = True,
    ) -> tuple:
        """Generate synthetic regression data.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            noise_level: Level of noise to add
            random_state: Random seed
            include_interactions: Whether to include feature interactions
            include_nonlinear: Whether to include nonlinear relationships

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        np.random.seed(random_state)

        # Generate features
        X = np.random.normal(0, 1, (n_samples, n_features))

        # Add nonlinear relationships
        if include_nonlinear:
            X[:, 0] = X[:, 0] ** 2  # Quadratic relationship
            X[:, 1] = np.sin(X[:, 1])  # Sinusoidal relationship
            if n_features > 2:
                X[:, 2] = np.exp(X[:, 2] * 0.5)  # Exponential relationship

        # Add interactions
        if include_interactions and n_features >= 4:
            X[:, 3] = X[:, 2] * X[:, 3]  # Interaction term

        # Generate target with noise
        true_weights = np.random.normal(0, 1, n_features)
        y = np.dot(X, true_weights) + noise_level * np.random.normal(0, 1, n_samples)

        # Normalize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)

        # Split data
        train_size = int(0.8 * n_samples)
        X_train = X_normalized[:train_size]
        X_test = X_normalized[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def generate_classification_data(
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 2,
        noise_level: float = 0.1,
        include_interactions: bool = True,
        include_nonlinear: bool = True,
        random_state: int = 42,
        sparse_features: bool = True,
        sparse_ratio: float = 0.3,
    ) -> tuple:
        """Generate synthetic classification data.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            noise_level: Level of noise to add
            include_interactions: Whether to include feature interactions
            include_nonlinear: Whether to include nonlinear relationships
            random_state: Random seed
            sparse_features: Whether to create sparse features
            sparse_ratio: Ratio of features that are relevant

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        np.random.seed(random_state)

        # Generate features
        X = np.random.normal(0, 1, (n_samples, n_features))

        # Add nonlinear relationships
        if include_nonlinear:
            X[:, 0] = X[:, 0] ** 2  # Quadratic relationship
            X[:, 1] = np.sin(X[:, 1])  # Sinusoidal relationship
            if n_features > 2:
                X[:, 2] = np.exp(X[:, 2] * 0.5)  # Exponential relationship

        # Add interactions
        if include_interactions and n_features >= 4:
            X[:, 3] = X[:, 2] * X[:, 3]  # Interaction term

        # Create sparse features if requested
        if sparse_features:
            sparse_mask = np.random.random(n_features) < sparse_ratio
            X_sparse = X.copy()
            X_sparse[:, ~sparse_mask] = 0
            X = X_sparse
        else:
            sparse_mask = np.ones(n_features, dtype=bool)  # All features are relevant

        # Create decision boundary
        if n_classes == 2:
            # Binary classification
            relevant_features = X[:, sparse_mask] if sparse_features else X
            decision_boundary = np.sum(relevant_features, axis=1) + 0.5 * np.sum(
                relevant_features**2,
                axis=1,
            )
            decision_boundary += noise_level * np.random.normal(0, 1, n_samples)
            y = (decision_boundary > np.median(decision_boundary)).astype(int)
        else:
            # Multi-class classification
            centers = np.random.normal(0, 2, (n_classes, n_features))
            y = np.zeros(n_samples)
            for i in range(n_samples):
                distances = [np.linalg.norm(X[i] - center) for center in centers]
                y[i] = np.argmin(distances)

        # Normalize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)

        # Split data
        train_size = int(0.8 * n_samples)
        X_train = X_normalized[:train_size]
        X_test = X_normalized[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def generate_anomaly_detection_data(
        n_normal: int = 1000,
        n_anomalies: int = 50,
        n_features: int = 50,
        random_state: int = 42,
        anomaly_type: str = "outlier",
    ) -> tuple:
        """Generate synthetic anomaly detection data.

        Args:
            n_normal: Number of normal samples
            n_anomalies: Number of anomaly samples
            n_features: Number of features
            random_state: Random seed
            anomaly_type: Type of anomalies ("outlier", "cluster", "drift")

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        np.random.seed(random_state)

        # Generate normal data (clustered)
        centers = [np.random.normal(0, 2, n_features) for _ in range(3)]
        normal_data = []
        for center in centers:
            cluster_data = np.random.normal(center, 1.0, (n_normal // 3, n_features))
            normal_data.append(cluster_data)

        # Add remaining samples to the last center
        remaining = n_normal - len(normal_data) * (n_normal // 3)
        if remaining > 0:
            last_center = centers[-1]
            remaining_data = np.random.normal(last_center, 1.0, (remaining, n_features))
            normal_data.append(remaining_data)

        normal_data_array = (
            np.vstack(normal_data)
            if normal_data
            else np.array([]).reshape(0, n_features)
        )

        # Generate anomaly data
        if anomaly_type == "outlier":
            anomaly_data = np.random.uniform(-10, 10, (n_anomalies, n_features))
        elif anomaly_type == "cluster":
            anomaly_center = np.random.normal(0, 5, n_features)
            anomaly_data = np.random.normal(
                anomaly_center,
                0.5,
                (n_anomalies, n_features),
            )
        elif anomaly_type == "drift":
            # Drift: same distribution but shifted
            drift_center = np.random.normal(3, 1, n_features)
            anomaly_data = np.random.normal(
                drift_center,
                1.0,
                (n_anomalies, n_features),
            )
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

        # Combine data
        all_data = np.vstack([normal_data_array, anomaly_data])
        labels = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

        # Normalize data
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        scaled_data = (all_data - mean) / (std + 1e-8)

        # Split data
        train_size = int(0.8 * len(scaled_data))
        X_train = scaled_data[:train_size]
        X_test = scaled_data[train_size:]
        y_train = labels[:train_size]
        y_test = labels[train_size:]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def generate_context_data(
        n_samples: int = 1500,
        n_features: int = 15,
        n_context: int = 8,
        random_state: int = 42,
        context_effect: float = 0.3,
    ) -> tuple:
        """Generate synthetic data with context information.

        Args:
            n_samples: Number of samples
            n_features: Number of main features
            n_context: Number of context features
            random_state: Random seed
            context_effect: Strength of context effect

        Returns:
            Tuple containing (X_train, X_test, context_train, context_test, y_train, y_test)
        """
        np.random.seed(random_state)

        # Generate main features
        X = np.random.normal(0, 1, (n_samples, n_features))

        # Generate context features (different distribution)
        context = np.random.uniform(-2, 2, (n_samples, n_context))

        # Create complex target that depends on both features and context
        context_weights = np.random.normal(0, 1, n_context)
        feature_weights = np.random.normal(0, 1, n_features)

        # Create context-dependent decision boundary
        context_effect_val = np.dot(context, context_weights)
        feature_effect = np.dot(X, feature_weights)
        interaction_effect = context_effect * np.sum(X[:, :5] * context[:, :5], axis=1)

        # Combine effects
        decision_boundary = feature_effect + context_effect_val + interaction_effect
        y = (decision_boundary > np.median(decision_boundary)).astype(int)

        # Normalize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)

        context_mean = np.mean(context, axis=0)
        context_std = np.std(context, axis=0)
        context_normalized = (context - context_mean) / (context_std + 1e-8)

        # Split data
        train_size = int(0.8 * n_samples)
        X_train = X_normalized[:train_size]
        X_test = X_normalized[train_size:]
        context_train = context_normalized[:train_size]
        context_test = context_normalized[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        return X_train, X_test, context_train, context_test, y_train, y_test

    @staticmethod
    def generate_multi_input_data(
        n_samples: int = 1000,
        feature_shapes: dict[str, tuple[int, ...]] = None,
        random_state: int = 42,
        task_type: str = "regression",
    ) -> tuple:
        """Generate multi-input data for preprocessing model testing.

        Args:
            n_samples: Number of samples
            feature_shapes: Dictionary mapping feature names to shapes
            random_state: Random seed
            task_type: Type of task - "regression" or "classification"

        Returns:
            Tuple containing (X_train_dict, X_test_dict, y_train, y_test)
        """
        if feature_shapes is None:
            feature_shapes = {"feature1": (20,), "feature2": (15,), "feature3": (10,)}

        np.random.seed(random_state)

        X_train_dict = {}
        X_test_dict = {}

        # Generate data for each feature
        for feature_name, shape in feature_shapes.items():
            # Generate random data with different distributions for each feature
            if "feature1" in feature_name:
                data = np.random.normal(0, 1, (n_samples,) + shape)
            elif "feature2" in feature_name:
                data = np.random.uniform(-2, 2, (n_samples,) + shape)
            else:
                data = np.random.exponential(1, (n_samples,) + shape)

            # Normalize
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            data_normalized = (data - data_mean) / (data_std + 1e-8)

            # Split
            train_size = int(0.8 * n_samples)
            X_train_dict[feature_name] = data_normalized[:train_size]
            X_test_dict[feature_name] = data_normalized[train_size:]

        # Generate target based on combined features (use full dataset before splitting)
        combined_features = np.concatenate(
            [
                np.vstack([X_train_dict[name], X_test_dict[name]])
                for name in feature_shapes.keys()
            ],
            axis=1,
        )
        target_weights = np.random.normal(0, 1, combined_features.shape[1])
        y = np.dot(combined_features, target_weights) + 0.1 * np.random.normal(
            0,
            1,
            combined_features.shape[0],
        )

        # Convert to classification if requested
        if task_type == "classification":
            y = (y > np.median(y)).astype(int)

        # Split target
        train_size = int(0.8 * n_samples)
        y_train = y[:train_size]
        y_test = y[train_size:]

        return X_train_dict, X_test_dict, y_train, y_test

    @staticmethod
    def create_preprocessing_model(
        input_shapes: dict[str, tuple[int, ...]],
        output_dim: int = 32,
        name: str = "preprocessing_model",
    ) -> keras.Model:
        """Create a preprocessing model for multi-input data.

        Args:
            input_shapes: Dictionary mapping input names to shapes
            output_dim: Output dimension
            name: Model name

        Returns:
            Keras preprocessing model
        """
        # Create input layers
        inputs = {}
        processed_inputs = []

        for input_name, input_shape in input_shapes.items():
            inputs[input_name] = keras.layers.Input(shape=input_shape, name=input_name)

            # Process each input
            if len(input_shape) == 1:
                # 1D input - use dense layers
                x = keras.layers.Dense(16, activation="relu")(inputs[input_name])
                x = keras.layers.Dropout(0.1)(x)
                x = keras.layers.Dense(16, activation="relu")(x)
            else:
                # Multi-dimensional input - use flatten + dense
                x = keras.layers.Flatten()(inputs[input_name])
                x = keras.layers.Dense(32, activation="relu")(x)
                x = keras.layers.Dropout(0.1)(x)
                x = keras.layers.Dense(16, activation="relu")(x)

            processed_inputs.append(x)

        # Combine processed inputs
        if len(processed_inputs) > 1:
            combined = keras.layers.Concatenate()(processed_inputs)
        else:
            combined = processed_inputs[0]

        # Final processing
        output = keras.layers.Dense(output_dim, activation="relu")(combined)
        output = keras.layers.Dropout(0.1)(output)

        # Create model
        model = keras.Model(inputs=inputs, outputs=output, name=name)

        return model

    @staticmethod
    def create_dataset(
        X: Union[np.ndarray, dict[str, np.ndarray]],
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> "tf.data.Dataset":
        """Create a TensorFlow dataset from data (requires TensorFlow).

        Args:
            X: Input data (array or dict of arrays)
            y: Target data
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            TensorFlow dataset (if TensorFlow is available)

        Raises:
            ImportError: If TensorFlow is not installed
        """
        if not _TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for create_dataset. "
                "Install it with: pip install tensorflow",
            )

        if isinstance(X, dict):
            # Multi-input data
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
        else:
            # Single input data
            dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(y))

        dataset = dataset.batch(batch_size)

        return dataset

    @staticmethod
    def generate_timeseries_data(
        n_samples: int = 1000,
        seq_len: int = 96,
        pred_len: int = 12,
        n_features: int = 7,
        random_state: int = 42,
        include_trend: bool = True,
        include_seasonality: bool = True,
        trend_direction: str = "up",
        noise_level: float = 0.1,
        scale: float = 1.0,
    ) -> tuple:
        """Generate synthetic multivariate time series data for forecasting.

        This method generates realistic time series data with optional trend and
        seasonality patterns, suitable for testing time series models like TSMixer
        and TimeMixer.

        Args:
            n_samples: Number of time series samples to generate.
            seq_len: Length of input sequence (lookback window).
            pred_len: Length of prediction horizon (forecast window).
            n_features: Number of time series features (channels).
            random_state: Random seed for reproducibility.
            include_trend: Whether to include trend component.
            include_seasonality: Whether to include seasonal component.
            trend_direction: Direction of trend ('up', 'down', 'random').
            noise_level: Standard deviation of Gaussian noise.
            scale: Scaling factor for the generated data.

        Returns:
            Tuple of (X, y) where:
            - X: Input sequences of shape (n_samples, seq_len, n_features)
            - y: Target sequences of shape (n_samples, pred_len, n_features)

        Example:
            >>> X, y = KerasFactoryDataGenerator.generate_timeseries_data(
            ...     n_samples=100,
            ...     seq_len=96,
            ...     pred_len=12,
            ...     n_features=7
            ... )
            >>> X.shape
            (100, 96, 7)
            >>> y.shape
            (100, 12, 7)
        """
        np.random.seed(random_state)

        total_len = seq_len + pred_len
        X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        y = np.zeros((n_samples, pred_len, n_features), dtype=np.float32)

        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                # Generate time steps
                t = np.arange(total_len)

                # Initialize time series
                ts = np.zeros(total_len)

                # Add trend
                if include_trend:
                    if trend_direction == "up":
                        trend = np.linspace(0, 1, total_len) * scale
                    elif trend_direction == "down":
                        trend = np.linspace(1, 0, total_len) * scale
                    else:  # random
                        trend_slope = np.random.uniform(-0.5, 0.5)
                        trend = trend_slope * t / total_len * scale
                    ts += trend

                # Add seasonality
                if include_seasonality:
                    seasonal_period = np.random.randint(7, 25)
                    seasonal_amplitude = np.random.uniform(0.2, 0.8) * scale
                    seasonality = seasonal_amplitude * np.sin(
                        2 * np.pi * t / seasonal_period,
                    )
                    ts += seasonality

                # Add base level
                base_level = np.random.uniform(-1, 1) * scale
                ts += base_level

                # Add noise
                noise = np.random.normal(0, noise_level, total_len)
                ts += noise

                # Split into input and target
                X[sample_idx, :, feature_idx] = ts[:seq_len]
                y[sample_idx, :, feature_idx] = ts[seq_len:]

        return X.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def generate_multivariate_timeseries(
        n_samples: int = 1000,
        seq_len: int = 96,
        pred_len: int = 12,
        n_features: int = 7,
        correlation_strength: float = 0.5,
        random_state: int = 42,
    ) -> tuple:
        """Generate correlated multivariate time series data.

        Creates time series where features have dependencies on each other,
        simulating real-world scenarios where different variables influence
        one another.

        Args:
            n_samples: Number of samples.
            seq_len: Input sequence length.
            pred_len: Prediction horizon.
            n_features: Number of features.
            correlation_strength: Strength of inter-feature correlations (0-1).
            random_state: Random seed.

        Returns:
            Tuple of (X, y) where X has shape (n_samples, seq_len, n_features)
            and y has shape (n_samples, pred_len, n_features).
        """
        np.random.seed(random_state)

        total_len = seq_len + pred_len
        X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        y = np.zeros((n_samples, pred_len, n_features), dtype=np.float32)

        # Generate correlation matrix
        if n_features > 1:
            # Create positive semi-definite correlation matrix
            L = np.random.randn(n_features, n_features)
            corr_matrix = correlation_strength * (L @ L.T)
            corr_matrix /= np.diag(corr_matrix).max()
            np.fill_diagonal(corr_matrix, 1.0)
        else:
            corr_matrix = np.array([[1.0]])

        for sample_idx in range(n_samples):
            # Generate independent noise
            noise = np.random.randn(total_len, n_features)

            # Apply correlation
            ts_data = (
                noise @ np.linalg.cholesky(corr_matrix + np.eye(n_features) * 0.01).T
            )

            # Add trend to first feature
            trend = np.linspace(0, 1, total_len)
            ts_data[:, 0] += trend

            # Split
            X[sample_idx, :, :] = ts_data[:seq_len]
            y[sample_idx, :, :] = ts_data[seq_len:]

        return X.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def generate_seasonal_timeseries(
        n_samples: int = 1000,
        seq_len: int = 96,
        pred_len: int = 12,
        n_features: int = 7,
        seasonal_period: int = 12,
        random_state: int = 42,
    ) -> tuple:
        """Generate strongly seasonal time series data.

        Ideal for testing decomposition-based models like TimeMixer that
        explicitly handle trend and seasonal components.

        Args:
            n_samples: Number of samples.
            seq_len: Input sequence length.
            pred_len: Prediction horizon.
            n_features: Number of features.
            seasonal_period: Period of seasonality.
            random_state: Random seed.

        Returns:
            Tuple of (X, y).
        """
        np.random.seed(random_state)

        total_len = seq_len + pred_len
        X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        y = np.zeros((n_samples, pred_len, n_features), dtype=np.float32)

        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                t = np.arange(total_len)

                # Base trend (slowly changing)
                base_trend = 50 + 20 * np.sin(2 * np.pi * t / (total_len * 2))

                # Strong seasonality
                seasonal = 10 * np.sin(2 * np.pi * t / seasonal_period)

                # Feature-specific cycle
                cycle = 5 * np.cos(2 * np.pi * (t / 30 + feature_idx / n_features))

                # Combine components
                ts = base_trend + seasonal + cycle

                # Add small noise
                ts += np.random.normal(0, 0.5, total_len)

                # Split
                X[sample_idx, :, feature_idx] = ts[:seq_len]
                y[sample_idx, :, feature_idx] = ts[seq_len:]

        return X.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def generate_anomalous_timeseries(
        n_samples: int = 1000,
        seq_len: int = 96,
        pred_len: int = 12,
        n_features: int = 7,
        anomaly_ratio: float = 0.1,
        anomaly_magnitude: float = 3.0,
        random_state: int = 42,
    ) -> tuple:
        """Generate time series with anomalies for anomaly detection testing.

        Args:
            n_samples: Number of samples.
            seq_len: Input sequence length.
            pred_len: Prediction horizon.
            n_features: Number of features.
            anomaly_ratio: Ratio of anomalous samples (0-1).
            anomaly_magnitude: Magnitude of anomalies in std deviations.
            random_state: Random seed.

        Returns:
            Tuple of (X, y, anomaly_labels) where anomaly_labels indicates
            which samples contain anomalies.
        """
        np.random.seed(random_state)

        # First generate normal data
        X, y = KerasFactoryDataGenerator.generate_timeseries_data(
            n_samples=n_samples,
            seq_len=seq_len,
            pred_len=pred_len,
            n_features=n_features,
            random_state=random_state,
        )

        # Create anomaly labels
        anomaly_labels = np.zeros(n_samples, dtype=np.int32)
        n_anomalies = int(n_samples * anomaly_ratio)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        anomaly_labels[anomaly_indices] = 1

        # Inject anomalies
        for idx in anomaly_indices:
            # Choose random anomaly type
            anomaly_type = np.random.choice(["spike", "drift", "noise"])

            for feature_idx in range(n_features):
                if anomaly_type == "spike":
                    # Sudden spike
                    spike_pos = np.random.randint(0, seq_len)
                    X[idx, spike_pos, feature_idx] += anomaly_magnitude * np.std(
                        X[:, spike_pos, feature_idx],
                    )

                elif anomaly_type == "drift":
                    # Gradual drift
                    drift = np.linspace(0, anomaly_magnitude, seq_len) * np.std(
                        X[:, :, feature_idx],
                    )
                    X[idx, :, feature_idx] += drift

                else:  # noise
                    # Excessive noise
                    noise = np.random.normal(0, anomaly_magnitude, seq_len) * np.std(
                        X[:, :, feature_idx],
                    )
                    X[idx, :, feature_idx] += noise

        return (
            X.astype(np.float32),
            y.astype(np.float32),
            anomaly_labels.astype(np.int32),
        )

    @staticmethod
    def generate_multiscale_timeseries(
        n_samples: int = 1000,
        seq_len: int = 96,
        pred_len: int = 12,
        n_features: int = 7,
        scales: list[int] | None = None,
        random_state: int = 42,
    ) -> tuple:
        """Generate multi-scale time series with components at different frequencies.

        Useful for testing models that use multi-scale mixing like TimeMixer.

        Args:
            n_samples: Number of samples.
            seq_len: Input sequence length.
            pred_len: Prediction horizon.
            n_features: Number of features.
            scales: List of frequency scales (default: [7, 14, 28, 56]).
            random_state: Random seed.

        Returns:
            Tuple of (X, y).
        """
        if scales is None:
            scales = [7, 14, 28, 56]

        np.random.seed(random_state)

        total_len = seq_len + pred_len
        X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        y = np.zeros((n_samples, pred_len, n_features), dtype=np.float32)

        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                t = np.arange(total_len)
                ts = np.zeros(total_len)

                # Add components at different scales
                for scale_idx, scale in enumerate(scales):
                    amplitude = 1.0 / (scale_idx + 1)  # Decreasing amplitude
                    ts += amplitude * np.sin(2 * np.pi * t / scale)

                # Add trend
                ts += 0.1 * t / total_len

                # Add noise
                ts += np.random.normal(0, 0.1, total_len)

                X[sample_idx, :, feature_idx] = ts[:seq_len]
                y[sample_idx, :, feature_idx] = ts[seq_len:]

        return X.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def generate_long_horizon_timeseries(
        n_samples: int = 500,
        seq_len: int = 336,
        pred_len: int = 336,
        n_features: int = 7,
        random_state: int = 42,
    ) -> tuple:
        """Generate long-horizon time series for testing long-term forecasting.

        Useful for benchmarking models on challenging long-range forecasting tasks.

        Args:
            n_samples: Number of samples.
            seq_len: Input sequence length (typically 336 = 2 weeks of hourly data).
            pred_len: Prediction horizon (typically 336 for long-horizon).
            n_features: Number of features.
            random_state: Random seed.

        Returns:
            Tuple of (X, y).
        """
        np.random.seed(random_state)

        total_len = seq_len + pred_len
        X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        y = np.zeros((n_samples, pred_len, n_features), dtype=np.float32)

        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                t = np.arange(total_len)

                # Weekly seasonality
                weekly = 10 * np.sin(2 * np.pi * t / 168)

                # Daily seasonality
                daily = 5 * np.sin(2 * np.pi * t / 24)

                # Long-term trend (very slow)
                long_trend = 2 * np.sin(2 * np.pi * t / (total_len * 4))

                # Combine
                ts = 50 + weekly + daily + long_trend

                # Add noise
                ts += np.random.normal(0, 0.3, total_len)

                X[sample_idx, :, feature_idx] = ts[:seq_len]
                y[sample_idx, :, feature_idx] = ts[seq_len:]

        return X.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def generate_synthetic_energy_demand(
        n_samples: int = 1000,
        seq_len: int = 168,
        pred_len: int = 24,
        n_features: int = 3,
        random_state: int = 42,
    ) -> tuple:
        """Generate synthetic energy demand time series.

        Simulates realistic energy consumption patterns with daily and weekly
        seasonality, useful for testing on realistic forecasting scenarios.

        Args:
            n_samples: Number of samples.
            seq_len: Input sequence length (default: 168 = 1 week).
            pred_len: Prediction horizon (default: 24 = 1 day).
            n_features: Number of features (e.g., residential, commercial, industrial).
            random_state: Random seed.

        Returns:
            Tuple of (X, y).
        """
        np.random.seed(random_state)

        total_len = seq_len + pred_len
        X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        y = np.zeros((n_samples, pred_len, n_features), dtype=np.float32)

        # Base demand for each sector
        base_demands = [100, 80, 50]  # Residential, Commercial, Industrial

        for sample_idx in range(n_samples):
            for feature_idx in range(min(n_features, len(base_demands))):
                t = np.arange(total_len)
                base = base_demands[feature_idx]

                # Daily pattern (peak during day, low at night)
                hour_of_day = t % 24
                daily_pattern = base * (
                    1 + 0.3 * np.sin(np.pi * hour_of_day / 12 - np.pi / 2)
                )

                # Weekly pattern (higher on weekdays)
                day_of_week = (t // 24) % 7
                weekly_pattern = 1 + 0.1 * np.cos(np.pi * day_of_week / 7)

                # Temperature effect (simplified)
                temp_effect = 5 * np.sin(2 * np.pi * t / total_len)

                # Combine
                demand = daily_pattern * weekly_pattern + temp_effect

                # Add noise
                demand += np.random.normal(0, 2, total_len)

                X[sample_idx, :, feature_idx] = np.maximum(demand[:seq_len], 0)
                y[sample_idx, :, feature_idx] = np.maximum(demand[seq_len:], 0)

            # For additional features, repeat with variations
            for feature_idx in range(len(base_demands), n_features):
                X[sample_idx, :, feature_idx] = X[
                    sample_idx,
                    :,
                    feature_idx % len(base_demands),
                ] * np.random.uniform(0.8, 1.2)
                y[sample_idx, :, feature_idx] = y[
                    sample_idx,
                    :,
                    feature_idx % len(base_demands),
                ] * np.random.uniform(0.8, 1.2)

        return X.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def create_timeseries_dataset(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> "tf.data.Dataset":
        """Create a TensorFlow dataset from time series data (requires TensorFlow).

        Args:
            X: Input sequences of shape (n_samples, seq_len, n_features).
            y: Target sequences of shape (n_samples, pred_len, n_features).
            batch_size: Batch size.
            shuffle: Whether to shuffle data.

        Returns:
            TensorFlow dataset with (X, y) pairs.

        Raises:
            ImportError: If TensorFlow is not installed

        Example:
            >>> X, y = KerasFactoryDataGenerator.generate_timeseries_data()
            >>> dataset = KerasFactoryDataGenerator.create_timeseries_dataset(X, y)
            >>> for x_batch, y_batch in dataset.take(1):
            ...     print(x_batch.shape, y_batch.shape)
        """
        if not _TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for create_timeseries_dataset. "
                "Install it with: pip install tensorflow",
            )

        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
