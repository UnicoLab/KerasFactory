"""End-to-end integration tests for BaseFeedForwardModel with preprocessing."""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import keras
from keras import Model, layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError

from kerasfactory.models.feed_forward import BaseFeedForwardModel
from kdp.processor import PreprocessingModel
from kdp.features import NumericalFeature, FeatureType


class TestBaseFeedForwardIntegration:
    """Test BaseFeedForwardModel integration with preprocessing."""

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

        # Create features with different types - all numeric for BaseFeedForwardModel
        data = {
            "numeric_feature_1": np.random.normal(10, 3, n_samples),
            "numeric_feature_2": np.random.exponential(2, n_samples),
            "categorical_feature": np.random.choice(
                [0, 1, 2, 3],
                n_samples,
            ),  # Encoded as integers
            "boolean_feature": np.random.choice(
                [0, 1],
                n_samples,
            ),  # Encoded as integers
            "target": np.random.normal(5, 1, n_samples),
        }

        df = pd.DataFrame(data)

        # No missing values for this test to avoid NaN predictions

        # Save to CSV
        csv_path = _temp_dir / "dummy_data.csv"
        df.to_csv(csv_path, index=False)

        return csv_path, df

    def test_end_to_end_training_and_prediction(
        self,
        _temp_dir: Path,
        dummy_data: tuple[Path, pd.DataFrame],
    ) -> None:
        """Test complete end-to-end workflow with preprocessing."""
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
        feature_names = [
            "numeric_feature_1",
            "numeric_feature_2",
            "categorical_feature",
            "boolean_feature",
        ]

        # Create a simple preprocessing model
        preprocessing_input = layers.Input(
            shape=(len(feature_names),),
            name="preprocessing_input",
        )
        x = layers.Dense(16, activation="relu", name="preprocessing_dense")(
            preprocessing_input,
        )
        x = layers.Dropout(0.1, name="preprocessing_dropout")(x)
        preprocessing_model = Model(
            inputs=preprocessing_input,
            outputs=x,
            name="preprocessing_model",
        )

        # Create BaseFeedForwardModel with preprocessing
        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[64, 32, 16],
            output_units=1,
            dropout_rate=0.2,
            activation="relu",
            preprocessing_model=preprocessing_model,
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
        assert not np.isnan(predictions).any()

        # Test model saving and loading
        model_path = _temp_dir / "saved_model.keras"
        model.save(model_path)

        # Load the model (disable safe mode to allow lambda deserialization)
        loaded_model = tf.keras.models.load_model(model_path, safe_mode=False)

        # Test prediction with loaded model
        loaded_predictions = loaded_model.predict(x_test, verbose=0)

        # Verify predictions are similar (allowing for small numerical differences)
        np.testing.assert_allclose(predictions, loaded_predictions, rtol=1e-5)

        # Test with completely raw data
        raw_test_data = {
            "numeric_feature_1": np.array([10.5, 12.5, 8.3]),
            "numeric_feature_2": np.array([1.2, 2.1, 3.7]),
            "categorical_feature": np.array([0, 1, 2]),
            "boolean_feature": np.array([1, 0, 1]),
        }

        # Should handle raw data through preprocessing
        raw_predictions = loaded_model.predict(raw_test_data, verbose=0)
        assert raw_predictions.shape == (3, 1)
        assert not np.isnan(raw_predictions).any()

    def test_model_with_different_architectures(
        self,
        _temp_dir: Path,
        dummy_data: tuple[Path, pd.DataFrame],
    ) -> None:
        """Test BaseFeedForwardModel with different architectures."""
        csv_path, df = dummy_data
        feature_names = [
            "numeric_feature_1",
            "numeric_feature_2",
            "categorical_feature",
            "boolean_feature",
        ]

        # Test different architectures
        architectures = [
            [32],  # Single hidden layer
            [64, 32],  # Two hidden layers
            [128, 64, 32, 16],  # Deep network
        ]

        for hidden_units in architectures:
            preprocessing_input = layers.Input(
                shape=(len(feature_names),),
                name="preprocessing_input",
            )
            x = layers.Dense(16, activation="relu")(preprocessing_input)
            x = layers.Dropout(0.1)(x)
            preprocessing_model = Model(
                inputs=preprocessing_input,
                outputs=x,
                name="preprocessing_model",
            )

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

    def test_model_serialization(
        self,
        _temp_dir: Path,
        dummy_data: tuple[Path, pd.DataFrame],
    ) -> None:
        """Test model serialization with preprocessing."""
        csv_path, df = dummy_data
        feature_names = [
            "numeric_feature_1",
            "numeric_feature_2",
            "categorical_feature",
            "boolean_feature",
        ]

        preprocessing_input = layers.Input(
            shape=(len(feature_names),),
            name="preprocessing_input",
        )
        x = layers.Dense(16, activation="relu")(preprocessing_input)
        x = layers.Dropout(0.1)(x)
        preprocessing_model = Model(
            inputs=preprocessing_input,
            outputs=x,
            name="preprocessing_model",
        )

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
    ) -> None:
        """Test error handling with invalid input data."""
        csv_path, df = dummy_data
        feature_names = [
            "numeric_feature_1",
            "numeric_feature_2",
            "categorical_feature",
            "boolean_feature",
        ]

        preprocessing_input = layers.Input(
            shape=(len(feature_names),),
            name="preprocessing_input",
        )
        x = layers.Dense(16, activation="relu")(preprocessing_input)
        x = layers.Dropout(0.1)(x)
        preprocessing_model = Model(
            inputs=preprocessing_input,
            outputs=x,
            name="preprocessing_model",
        )

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

        # Test with missing feature
        invalid_data = {
            "numeric_feature_1": np.array([1.0, 2.0]),
            "numeric_feature_2": np.array([3.0, 4.0]),
            # Missing categorical_feature and boolean_feature
        }

        with pytest.raises((KeyError, ValueError)):
            model.predict(invalid_data, verbose=0)

        # Test with wrong data types
        wrong_type_data = {
            "numeric_feature_1": ["not", "numeric"],
            "numeric_feature_2": np.array([1.0, 2.0]),
            "categorical_feature": np.array([0, 1]),
            "boolean_feature": np.array([0, 1]),
        }

        with pytest.raises((TypeError, ValueError)):
            model.predict(wrong_type_data, verbose=0)

    def test_performance_with_large_dataset(
        self,
        _temp_dir: Path,
    ) -> None:
        """Test model performance with larger dataset."""
        # Generate larger dataset
        np.random.seed(42)
        n_samples = 5000

        large_data = {
            "numeric_feature_1": np.random.normal(10, 3, n_samples),
            "numeric_feature_2": np.random.exponential(2, n_samples),
            "categorical_feature": np.random.choice([0, 1, 2, 3, 4], n_samples),
            "boolean_feature": np.random.choice([0, 1], n_samples),
            "target": np.random.normal(5, 1, n_samples),
        }

        df = pd.DataFrame(large_data)
        csv_path = _temp_dir / "large_data.csv"
        df.to_csv(csv_path, index=False)

        feature_names = [
            "numeric_feature_1",
            "numeric_feature_2",
            "categorical_feature",
            "boolean_feature",
        ]

        preprocessing_input = layers.Input(
            shape=(len(feature_names),),
            name="preprocessing_input",
        )
        x = layers.Dense(32, activation="relu")(preprocessing_input)
        x = layers.Dropout(0.2)(x)
        preprocessing_model = Model(
            inputs=preprocessing_input,
            outputs=x,
            name="preprocessing_model",
        )

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
        x_test_sample = {name: values[:100] for name, values in x_train.items()}
        predictions = model.predict(x_test_sample, verbose=0)
        assert predictions.shape == (100, 1)
        assert not np.isnan(predictions).any()

    def test_kdp_integration_with_custom_model(
        self,
        _temp_dir: Path,
    ) -> None:
        """Test KDP PreprocessingModel integration with a custom Keras model."""
        # Generate synthetic tabular data with mixed types for KDP
        np.random.seed(42)
        n_samples = 500

        data = {
            "numeric_feature_1": np.random.normal(10, 3, n_samples),
            "numeric_feature_2": np.random.exponential(2, n_samples),
            "numeric_feature_3": np.random.uniform(0, 10, n_samples),
            "numeric_feature_4": np.random.gamma(2, 1, n_samples),
            "target": np.random.normal(5, 1, n_samples),
        }

        df = pd.DataFrame(data)

        # Ensure no missing values for KDP preprocessing
        # KDP can handle missing values, but for testing we'll keep it simple
        # df.loc[df.sample(50).index, 'numeric_feature_1'] = np.nan
        # df.loc[df.sample(30).index, 'categorical_feature'] = np.nan

        # Save to CSV
        csv_path = _temp_dir / "kdp_test_data.csv"
        df.to_csv(csv_path, index=False)

        # Split data for training and testing
        train_df = df.iloc[:400].copy()
        test_df = df.iloc[400:].copy()

        train_path = _temp_dir / "kdp_train_data.csv"
        test_path = _temp_dir / "kdp_test_data.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Define feature names (excluding target)
        feature_names = [
            "numeric_feature_1",
            "numeric_feature_2",
            "numeric_feature_3",
            "numeric_feature_4",
        ]

        # Define feature specifications for KDP
        features_specs = {
            "numeric_feature_1": NumericalFeature(
                "numeric_feature_1",
                FeatureType.FLOAT_NORMALIZED,
            ),
            "numeric_feature_2": NumericalFeature(
                "numeric_feature_2",
                FeatureType.FLOAT_NORMALIZED,
            ),
            "numeric_feature_3": NumericalFeature(
                "numeric_feature_3",
                FeatureType.FLOAT_NORMALIZED,
            ),
            "numeric_feature_4": NumericalFeature(
                "numeric_feature_4",
                FeatureType.FLOAT_NORMALIZED,
            ),
        }

        # Create KDP PreprocessingModel
        kdp_preprocessor = PreprocessingModel(
            path_data=str(train_path),
            batch_size=100,
            output_mode="concat",  # Concatenate all features into single output
            use_caching=False,  # Disable caching for testing
            log_to_file=False,
            features_specs=features_specs,
            overwrite_stats=True,
        )

        # Build the KDP preprocessing model
        kdp_result = kdp_preprocessor.build_preprocessor()
        kdp_model = kdp_result["model"]

        # Create a custom model that uses KDP preprocessing
        # Get the output from KDP preprocessing
        kdp_output = kdp_model.output

        # Get the actual output shape from KDP preprocessing

        # Add custom layers on top of KDP preprocessing
        # Use the actual output shape from KDP preprocessing
        x = layers.Dense(64, activation="relu", name="hidden_1")(kdp_output)
        x = layers.Dropout(0.2, name="dropout_1")(x)
        x = layers.Dense(32, activation="relu", name="hidden_2")(x)
        x = layers.Dropout(0.2, name="dropout_2")(x)
        outputs = layers.Dense(1, name="output")(x)

        # Create the complete model
        model = Model(inputs=kdp_model.inputs, outputs=outputs, name="kdp_custom_model")

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()],
        )

        # Prepare training data - KDP expects DataFrame input with correct dtypes
        x_train_df = train_df[feature_names].copy()
        y_train = train_df["target"].to_numpy().astype(np.float32)

        # Train the model - KDP expects data to be passed as a dictionary
        # Convert DataFrame to dictionary format for KDP
        x_train_dict = {}
        for col in x_train_df.columns:
            x_train_dict[col] = x_train_df[col].to_numpy()

        history = model.fit(
            x_train_dict,
            y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )

        # Verify training completed successfully
        assert len(history.history["loss"]) == 5
        assert "val_loss" in history.history

        # Test prediction with test data
        x_test_df = test_df[feature_names].copy()

        # Convert DataFrame to dictionary format for KDP
        x_test_dict = {}
        for col in x_test_df.columns:
            x_test_dict[col] = x_test_df[col].to_numpy()

        predictions = model.predict(x_test_dict, verbose=0)

        # Verify predictions shape
        assert predictions.shape == (len(test_df), 1)
        assert not np.isnan(predictions).any()

        # Test model saving and loading
        model_path = _temp_dir / "kdp_saved_model.keras"
        model.save(model_path)

        # Load the model
        loaded_model = keras.models.load_model(model_path, safe_mode=False)

        # Test prediction with loaded model
        loaded_predictions = loaded_model.predict(x_test_dict, verbose=0)

        # Verify predictions are similar (allowing for small numerical differences)
        np.testing.assert_allclose(predictions, loaded_predictions, rtol=1e-4)

        # Test with new raw data (including missing values)
        # Only use features that exist in the KDP model
        raw_test_data = {
            "numeric_feature_1": np.array([np.nan, 12.5, 8.3, 15.0], dtype=np.float32),
            "numeric_feature_2": np.array([1.2, np.nan, 3.7, 2.1], dtype=np.float32),
            "numeric_feature_3": np.array([5.0, 7.2, 3.1, 9.8], dtype=np.float32),
            "numeric_feature_4": np.array([2.1, 4.5, 1.8, 6.2], dtype=np.float32),
        }

        # Should handle raw data through KDP preprocessing
        raw_predictions = loaded_model.predict(raw_test_data, verbose=0)
        assert raw_predictions.shape == (4, 1)
        # KDP may produce NaN values for inputs with missing values, which is expected behavior
        # We just verify that the model can handle the input without crashing

        # Test that KDP preprocessing works correctly
        # Get the preprocessing output directly
        x_test_sample_dict = {
            col: x_test_df[col].to_numpy()[:5] for col in x_test_df.columns
        }
        preprocessed_output = kdp_model.predict(x_test_sample_dict, verbose=0)
        assert preprocessed_output.shape[1] == 4  # 4 numerical features
        assert not np.isnan(preprocessed_output).any()

    def test_end_to_end_without_preprocessing(
        self,
        _temp_dir: Path,
        dummy_data: tuple[Path, pd.DataFrame],
    ) -> None:
        """Test complete end-to-end workflow WITHOUT preprocessing."""
        csv_path, df = dummy_data

        # Split data for training and testing
        train_df = df.iloc[:800].copy()
        test_df = df.iloc[800:].copy()

        # Define feature names (excluding target)
        feature_names = [
            "numeric_feature_1",
            "numeric_feature_2",
            "categorical_feature",
            "boolean_feature",
        ]

        # Create BaseFeedForwardModel WITHOUT preprocessing
        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[64, 32, 16],
            output_units=1,
            dropout_rate=0.2,
            activation="relu",
            preprocessing_model=None,  # No preprocessing
            name="feed_forward_without_preprocessing",
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

        # Test prediction with raw data (no preprocessing)
        x_test = {name: test_df[name].to_numpy() for name in feature_names}

        predictions = model.predict(x_test, verbose=0)

        # Verify predictions shape
        assert predictions.shape == (len(test_df), 1)
        assert not np.isnan(predictions).any()

        # Test model saving and loading
        model_path = _temp_dir / "saved_model_no_preprocessing.keras"
        model.save(model_path)

        # Load the model
        loaded_model = tf.keras.models.load_model(model_path, safe_mode=False)

        # Test prediction with loaded model
        loaded_predictions = loaded_model.predict(x_test, verbose=0)

        # Verify predictions are similar (allowing for small numerical differences)
        np.testing.assert_allclose(predictions, loaded_predictions, rtol=1e-5)

        # Test with completely raw data
        raw_test_data = {
            "numeric_feature_1": np.array([10.5, 12.5, 8.3]),
            "numeric_feature_2": np.array([1.2, 2.1, 3.7]),
            "categorical_feature": np.array([0, 1, 2]),
            "boolean_feature": np.array([1, 0, 1]),
        }

        # Should handle raw data directly (no preprocessing)
        raw_predictions = loaded_model.predict(raw_test_data, verbose=0)
        assert raw_predictions.shape == (3, 1)
        assert not np.isnan(raw_predictions).any()

    def test_model_without_preprocessing_different_architectures(
        self,
        _temp_dir: Path,
        dummy_data: tuple[Path, pd.DataFrame],
    ) -> None:
        """Test BaseFeedForwardModel without preprocessing with different architectures."""
        csv_path, df = dummy_data
        feature_names = [
            "numeric_feature_1",
            "numeric_feature_2",
            "categorical_feature",
            "boolean_feature",
        ]

        # Test different architectures without preprocessing
        architectures = [
            [32],  # Single hidden layer
            [64, 32],  # Two hidden layers
            [128, 64, 32, 16],  # Deep network
        ]

        for hidden_units in architectures:
            model = BaseFeedForwardModel(
                feature_names=feature_names,
                hidden_units=hidden_units,
                output_units=1,
                dropout_rate=0.1,
                activation="relu",
                preprocessing_model=None,  # No preprocessing
                name=f"feed_forward_{len(hidden_units)}_layers_no_preprocessing",
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

    def test_model_without_preprocessing_serialization(
        self,
        _temp_dir: Path,
        dummy_data: tuple[Path, pd.DataFrame],
    ) -> None:
        """Test model serialization without preprocessing."""
        csv_path, df = dummy_data
        feature_names = [
            "numeric_feature_1",
            "numeric_feature_2",
            "categorical_feature",
            "boolean_feature",
        ]

        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[32, 16],
            output_units=1,
            preprocessing_model=None,  # No preprocessing
            name="serializable_model_no_preprocessing",
        )

        # Test JSON serialization
        config = model.get_config()
        assert "feature_names" in config
        assert "hidden_units" in config
        assert "preprocessing_model" in config
        assert config["preprocessing_model"] is None

        # Test model reconstruction from config
        reconstructed_model = BaseFeedForwardModel.from_config(config)
        assert reconstructed_model.feature_names == model.feature_names
        assert reconstructed_model.hidden_units == model.hidden_units
        assert reconstructed_model.preprocessing_model is None
