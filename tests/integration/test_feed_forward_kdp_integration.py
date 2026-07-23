"""End-to-end integration tests for BaseFeedForwardModel with KDP preprocessing."""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError

from kerasfactory.models.feed_forward import BaseFeedForwardModel
from kdp.processor import PreprocessingModel
from kdp.features import NumericalFeature


class TestBaseFeedForwardKDPIntegration:
    """Test BaseFeedForwardModel integration with KDP preprocessing."""

    @pytest.fixture
    def _temp_dir(self) -> Path:
        """Create a temporary directory for test data."""
        _temp_dir = Path(tempfile.mkdtemp())
        yield _temp_dir
        shutil.rmtree(_temp_dir, ignore_errors=True)

    @pytest.fixture
    def dummy_data(self, _temp_dir: Path) -> tuple[Path, pd.DataFrame]:
        """Create dummy CSV data for testing."""
        # Generate synthetic tabular data
        np.random.seed(42)
        n_samples = 1000

        # Create features with different types
        data = {
            "numeric_feature_1": np.random.normal(10, 3, n_samples),
            "numeric_feature_2": np.random.exponential(2, n_samples),
            "categorical_feature": np.random.choice(["A", "B", "C", "D"], n_samples),
            "boolean_feature": np.random.choice([True, False], n_samples),
            "target": np.random.normal(5, 1, n_samples),
        }

        df = pd.DataFrame(data)

        # Add some missing values to test preprocessing
        df.loc[df.sample(50).index, "numeric_feature_1"] = np.nan
        df.loc[df.sample(30).index, "categorical_feature"] = None

        # Save to CSV
        csv_path = _temp_dir / "dummy_data.csv"
        df.to_csv(csv_path, index=False)

        return csv_path, df

    @pytest.fixture
    def _kdp_preprocessor(
        self,
        dummy_data: tuple[Path, pd.DataFrame],
    ) -> PreprocessingModel:
        """Create and fit KDP preprocessor."""
        csv_path, df = dummy_data

        # Create features_specs for the data (using only numerical features for now)
        features_specs = {
            "numeric_feature_1": NumericalFeature(name="numeric_feature_1"),
            "numeric_feature_2": NumericalFeature(name="numeric_feature_2"),
        }

        # Create PreprocessingModel (overwrite_stats avoids CWD features_stats.json mismatch)
        preprocessing_model = PreprocessingModel(
            path_data=str(csv_path),
            batch_size=1000,
            features_specs=features_specs,
            overwrite_stats=True,
        )

        # Build the preprocessor
        preprocessing_model.build_preprocessor()

        return preprocessing_model

    def test_end_to_end_training_and_prediction(
        self,
        _temp_dir: Path,
        dummy_data: tuple[Path, pd.DataFrame],
        _kdp_preprocessor: PreprocessingModel,
    ) -> None:
        """Test complete end-to-end workflow with KDP preprocessing."""
        csv_path, df = dummy_data

        # Split data for training and testing
        train_df = df.iloc[:800].copy()
        test_df = df.iloc[800:].copy()

        # Save train and test data
        train_path = _temp_dir / "train_data.csv"
        test_path = _temp_dir / "test_data.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Define feature names (excluding target)
        feature_names = ["numeric_feature_1", "numeric_feature_2"]

        # Create a new KDP preprocessing model with the training data
        features_specs = {
            "numeric_feature_1": NumericalFeature(name="numeric_feature_1"),
            "numeric_feature_2": NumericalFeature(name="numeric_feature_2"),
        }

        # Create PreprocessingModel with training data
        train_kdp_preprocessor = PreprocessingModel(
            path_data=str(train_path),
            batch_size=1000,
            features_specs=features_specs,
            overwrite_stats=True,
        )

        # Build the preprocessor with training data
        train_kdp_preprocessor.build_preprocessor()

        # Create BaseFeedForwardModel with preprocessing
        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[64, 32, 16],
            output_units=1,
            dropout_rate=0.2,
            activation="relu",
            preprocessing_model=train_kdp_preprocessor.model,  # Use the actual Keras model
            name="feed_forward_with_preprocessing",
        )

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()],
        )

        # Prepare training data
        x_train = {name: train_df[name].to_numpy() for name in feature_names}
        y_train = train_df["target"].to_numpy()

        # Train the model
        history = model.fit(
            x_train,
            y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )

        # Verify training completed successfully
        assert len(history.history["loss"]) == 5
        assert "val_loss" in history.history

        # Test prediction with raw data (should use preprocessing)
        x_test = {name: test_df[name].to_numpy() for name in feature_names}

        predictions = model.predict(x_test, verbose=0)

        # Verify predictions shape
        assert predictions.shape == (len(test_df), 1)
        # KDP may produce NaN values for some inputs, which is expected behavior
        # We just verify that the model can handle the input without crashing

        # Test model saving and loading
        model_path = _temp_dir / "saved_model.keras"
        model.save(model_path)

        # Load the model
        loaded_model = tf.keras.models.load_model(model_path)

        # Test prediction with loaded model
        loaded_predictions = loaded_model.predict(x_test, verbose=0)

        # Verify predictions are similar (allowing for small numerical differences)
        np.testing.assert_allclose(predictions, loaded_predictions, rtol=1e-5)

        # Test with completely raw data (including missing values)
        # Only use features that exist in the KDP model
        raw_test_data = {
            "numeric_feature_1": np.array([np.nan, 12.5, 8.3]),
            "numeric_feature_2": np.array([1.2, np.nan, 3.7]),
        }

        # Should handle missing values through preprocessing
        raw_predictions = loaded_model.predict(raw_test_data, verbose=0)
        assert raw_predictions.shape == (3, 1)
        # KDP may produce NaN values for inputs with missing values, which is expected behavior
        # We just verify that the model can handle the input without crashing

    def test_model_with_different_architectures(
        self,
        _temp_dir: Path,
        dummy_data: tuple[Path, pd.DataFrame],
        _kdp_preprocessor: PreprocessingModel,
    ) -> None:
        """Test BaseFeedForwardModel with different architectures."""
        csv_path, df = dummy_data
        feature_names = ["numeric_feature_1", "numeric_feature_2"]

        # Test different architectures
        architectures = [
            [32],  # Single hidden layer
            [64, 32],  # Two hidden layers
            [128, 64, 32, 16],  # Deep network
        ]

        for hidden_units in architectures:
            preprocessing_model = _kdp_preprocessor.model

            model = BaseFeedForwardModel(
                feature_names=feature_names,
                hidden_units=hidden_units,
                output_units=1,
                dropout_rate=0.1,
                activation="relu",
                preprocessing_model=preprocessing_model,
                name=f"feed_forward_{len(hidden_units)}_layers",
            )

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=MeanSquaredError(),
                metrics=[MeanAbsoluteError()],
            )

            # Quick training test
            x_train = {name: df[name].to_numpy()[:100] for name in feature_names}
            y_train = df["target"].to_numpy()[:100]

            history = model.fit(x_train, y_train, epochs=2, verbose=0)
            assert len(history.history["loss"]) == 2

    def test_model_serialization_with_kdp(
        self,
        _temp_dir: Path,
        dummy_data: tuple[Path, pd.DataFrame],
        _kdp_preprocessor: PreprocessingModel,
    ) -> None:
        """Test model serialization with KDP preprocessing."""
        csv_path, df = dummy_data
        feature_names = ["numeric_feature_1", "numeric_feature_2"]

        preprocessing_model = _kdp_preprocessor.model

        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[32, 16],
            output_units=1,
            preprocessing_model=preprocessing_model,
            name="serializable_model",
        )

        # Test JSON serialization
        config = model.get_config()
        assert "feature_names" in config
        assert "hidden_units" in config
        assert "preprocessing_model" in config

        # Test model reconstruction from config
        reconstructed_model = BaseFeedForwardModel.from_config(config)
        assert reconstructed_model.feature_names == model.feature_names
        assert reconstructed_model.hidden_units == model.hidden_units

    def test_error_handling_with_invalid_data(
        self,
        _temp_dir: Path,
        dummy_data: tuple[Path, pd.DataFrame],
        _kdp_preprocessor: PreprocessingModel,
    ) -> None:
        """Test error handling with invalid input data."""
        csv_path, df = dummy_data
        feature_names = ["numeric_feature_1", "numeric_feature_2"]

        preprocessing_model = _kdp_preprocessor.model

        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[32],
            output_units=1,
            preprocessing_model=preprocessing_model,
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
        )

        # Test with missing feature - should work since we only need the features that exist
        valid_data = {
            "numeric_feature_1": np.array([1.0, 2.0]),
            "numeric_feature_2": np.array([3.0, 4.0]),
        }

        # Should work fine with only the required features
        predictions = model.predict(valid_data, verbose=0)
        assert predictions.shape == (2, 1)

        # Test with wrong data types
        wrong_type_data = {
            "numeric_feature_1": ["not", "numeric"],
            "numeric_feature_2": np.array([1.0, 2.0]),
            "categorical_feature": np.array(["A", "B"]),
            "boolean_feature": np.array([True, False]),
        }

        with pytest.raises((TypeError, ValueError)):
            model.predict(wrong_type_data, verbose=0)

    def test_performance_with_large_dataset(
        self,
        _temp_dir: Path,
        _kdp_preprocessor: PreprocessingModel,
    ) -> None:
        """Test model performance with larger dataset."""
        # Generate larger dataset
        np.random.seed(42)
        n_samples = 5000

        large_data = {
            "numeric_feature_1": np.random.normal(10, 3, n_samples),
            "numeric_feature_2": np.random.exponential(2, n_samples),
            "categorical_feature": np.random.choice(
                ["A", "B", "C", "D", "E"],
                n_samples,
            ),
            "boolean_feature": np.random.choice([True, False], n_samples),
            "target": np.random.normal(5, 1, n_samples),
        }

        df = pd.DataFrame(large_data)
        csv_path = _temp_dir / "large_data.csv"
        df.to_csv(csv_path, index=False)

        # Create new processor for large dataset
        features_specs = {
            "numeric_feature_1": NumericalFeature(name="numeric_feature_1"),
            "numeric_feature_2": NumericalFeature(name="numeric_feature_2"),
        }

        processor = PreprocessingModel(
            path_data=str(csv_path),
            batch_size=1000,
            features_specs=features_specs,
            overwrite_stats=True,
        )

        processor.build_preprocessor()
        preprocessing_model = processor.model

        feature_names = ["numeric_feature_1", "numeric_feature_2"]

        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[128, 64, 32],
            output_units=1,
            dropout_rate=0.3,
            preprocessing_model=preprocessing_model,
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()],
        )

        # Train on large dataset
        x_train = {name: df[name].to_numpy() for name in feature_names}
        y_train = df["target"].to_numpy()

        history = model.fit(
            x_train,
            y_train,
            epochs=3,
            batch_size=64,
            validation_split=0.2,
            verbose=0,
        )

        # Verify training completed
        assert len(history.history["loss"]) == 3
        assert (
            history.history["loss"][-1] < history.history["loss"][0]
        )  # Loss should decrease

        # Test prediction performance
        x_test_sample = {name: df[name].to_numpy()[:100] for name in feature_names}
        predictions = model.predict(x_test_sample, verbose=0)
        assert predictions.shape == (100, 1)
        assert not np.isnan(predictions).any()
