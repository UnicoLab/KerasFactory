# 🌟 KerasFactory - Reusable Model Architecture Bricks in Keras 🌟

<div align="center">
  <img src="docs/logo.png" width="350" alt="KerasFactory Logo"/>
  
  <p align="center"><strong>Provided and maintained by <a href="https://unicolab.ai">🦄 UnicoLab</a></strong></p>
</div>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Keras 3.8+](https://img.shields.io/badge/keras-3.8+-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![🦄 UnicoLab](https://img.shields.io/badge/UnicoLab-Enterprise%20AI-blue.svg)](https://unicolab.ai)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://unicolab.github.io/KerasFactory/)

**KerasFactory** is a comprehensive collection of reusable Keras layers and models specifically designed for tabular data processing, feature engineering, and advanced neural network architectures. Built with Keras 3 and developed by [🦄 UnicoLab](https://unicolab.ai), it provides a clean, efficient, and extensible foundation for building sophisticated machine learning models for enterprise AI applications.

## ✨ Key Features

- **🎯 38+ Production-Ready Layers**: Attention mechanisms, feature processing, preprocessing, and specialized architectures
- **🧠 Advanced Models**: SFNE blocks, Terminator models, and more coming soon
- **📊 Data Analyzer**: Intelligent CSV analysis tool that recommends appropriate layers
- **🔬 Experimental Modules**: 20+ cutting-edge layers and models for research
- **⚡ Keras 3 Only**: Pure Keras 3 implementation with no TensorFlow dependencies
- **🧪 Comprehensive Testing**: Full test coverage with 38+ test suites
- **📚 Rich Documentation**: Detailed guides, examples, and API documentation

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
poetry add kerasfactory

# Or install from source
git clone https://github.com/UnicoLab/KerasFactory
cd KerasFactory
poetry install
```

### 🚀 Quick Start Examples

#### Example 1: Smart Data Preprocessing

```python
import keras
from kerasfactory.layers import DistributionTransformLayer

# Create a simple model with automatic data transformation
inputs = keras.Input(shape=(10,))  # 10 numerical features

# Automatically transform data to normal distribution
transformed = DistributionTransformLayer(transform_type='auto')(inputs)

# Simple neural network
x = keras.layers.Dense(64, activation='relu')(transformed)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Model ready! The layer will automatically choose the best transformation for your data.")
```

#### Example 2: Intelligent Feature Fusion

```python
import keras
from kerasfactory.layers import GatedFeatureFusion

# Create two different representations of your data
inputs = keras.Input(shape=(8,))  # 8 features

# First representation: linear processing
linear_features = keras.layers.Dense(16, activation='relu')(inputs)

# Second representation: non-linear processing  
nonlinear_features = keras.layers.Dense(16, activation='tanh')(inputs)

# Intelligently combine both representations
fused_features = GatedFeatureFusion()([linear_features, nonlinear_features])

# Final prediction
outputs = keras.layers.Dense(1, activation='sigmoid')(fused_features)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy')

print("Smart feature fusion model ready! The layer learns which representation to trust more.")
```

#### Example 3: Ready-to-Use Models

```python
import keras
from kerasfactory.models import BaseFeedForwardModel

# Create a complete model with just one line!
model = BaseFeedForwardModel(
    feature_names=['age', 'income', 'education', 'experience'],
    hidden_units=[64, 32, 16],
    output_units=1,
    dropout_rate=0.2
)

# Your data (each feature as separate input)
age = keras.random.normal((100, 1))
income = keras.random.normal((100, 1)) 
education = keras.random.normal((100, 1))
experience = keras.random.normal((100, 1))

# Train with one command
model.compile(optimizer='adam', loss='mse')
model.fit([age, income, education, experience], 
          keras.random.normal((100, 1)), 
          epochs=10, verbose=0)

print("✅ Model trained successfully! No complex setup needed.")
```

#### Example 4: Date Feature Engineering

```python
import keras
from kerasfactory.layers import DateEncodingLayer

# Create a model that processes date information
inputs = keras.Input(shape=(4,))  # [year, month, day, day_of_week]

# Convert dates to cyclical features automatically
date_features = DateEncodingLayer()(inputs)

# Simple prediction model
x = keras.layers.Dense(32, activation='relu')(date_features)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy')

print("📅 Date-aware model ready! Handles seasonality and cyclical patterns automatically.")
```

### 🧠 Smart Data Analyzer

```python
from kerasfactory.utils import analyze_data

# Get intelligent recommendations for your data
results = analyze_data("your_data.csv")
recommendations = results["recommendations"]

print("🎯 Recommended layers for your data:")
for layer in recommendations:
    print(f"  • {layer['layer_name']}: {layer['description']}")
    
print("✨ No more guessing which layers to use!")
```

## 🏗️ Architecture Overview

### Core Components

#### **Layers** (`kerasfactory.layers`)
- **Attention Mechanisms**: `TabularAttention`, `MultiResolutionTabularAttention`, `ColumnAttention`, `RowAttention`
- **Feature Processing**: `AdvancedNumericalEmbedding`, `GatedFeatureFusion`, `VariableSelection`
- **Preprocessing**: `DateEncodingLayer`, `DateParsingLayer`, `DifferentiableTabularPreprocessor`
- **Advanced Architectures**: `TransformerBlock`, `GatedResidualNetwork`, `BoostingBlock`
- **Specialized Layers**: `BusinessRulesLayer`, `StochasticDepth`, `FeatureCutout`

#### **Models** (`kerasfactory.models`)
- **SFNEBlock**: Advanced feature processing block
- **TerminatorModel**: Multi-block hierarchical processing model

#### **Utilities** (`kerasfactory.utils`)
- **Data Analyzer**: Intelligent CSV analysis and layer recommendation system
- **CLI Tools**: Command-line interface for data analysis

#### **Experimental** (`experimental/`)
- **Time Series**: 12+ specialized time series preprocessing layers
- **Advanced Models**: Neural Additive Models, Temporal Fusion Transformers, and more
- **Research Components**: Cutting-edge architectures for experimentation
- **Note**: Experimental components are not included in the PyPI package

## 📖 Documentation

- **[Online Documentation](https://unicolab.github.io/KerasFactory/)**: Full API reference with automatic docstring generation
- **[API Reference](https://unicolab.github.io/KerasFactory/api/)**: Complete documentation for all layers, models, and utilities
- **[Layer Implementation Guide](docs/layers_implementation_guide.md)**: Comprehensive guide for implementing new layers
- **[Data Analyzer Documentation](docs/data_analyzer.md)**: Complete guide to the data analysis tools
- **[Contributing Guide](docs/contributing.md)**: How to contribute to the project

## 🎯 Common Use Cases

### 📊 Tabular Data Processing
```python
from kerasfactory.layers import DistributionTransformLayer, GatedFeatureFusion

# Smart preprocessing
preprocessor = DistributionTransformLayer(transform_type='auto')

# Feature combination
fusion = GatedFeatureFusion()
```

### 🔧 Feature Engineering
```python
from kerasfactory.layers import DateEncodingLayer, BusinessRulesLayer

# Date features
date_encoder = DateEncodingLayer()

# Business rules validation
rules = BusinessRulesLayer(
    rules=[(">", 0), ("<", 100)], 
    feature_type="numerical"
)
```

### 🎨 Advanced Architectures
```python
from kerasfactory.layers import StochasticDepth, GatedResidualNetwork

# Regularization
stochastic_depth = StochasticDepth(survival_prob=0.8)

# Advanced processing
grn = GatedResidualNetwork(units=64)
```

## 🧪 Testing

```bash
# Run all tests
make all_tests

# Run specific test categories
make unittests
make data_analyzer_tests

# Generate coverage report
make coverage
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/UnicoLab/KerasFactory.git
cd KerasFactory

# Install development dependencies
poetry install

# Install pre-commit hooks
pre-commit install

# Run tests
make all_tests
```

### Commit Convention

We use semantic commit messages:
- `feat(KerasFactory): add new layer for feature processing`
- `fix(KerasFactory): resolve serialization issue`
- `docs(KerasFactory): update installation guide`

## 📊 Performance

KerasFactory is optimized for performance with:
- **Keras 3 Backend**: Leverages the latest Keras optimizations
- **Efficient Operations**: Uses only Keras operations for maximum compatibility
- **Memory Optimization**: Careful memory management in complex layers
- **Batch Processing**: Optimized for batch operations

## 💬 Join Our Community

Have questions or want to connect with other KDP users? Join us on Discord:

[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.gg/6zf4VZFYV5)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Keras 3](https://keras.io/)
- Inspired by modern deep learning research
- Community-driven development

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/UnicoLab/KerasFactory/issues)
- **Discussions**: [GitHub Discussions](https://github.com/UnicoLab/KerasFactory/discussions)
- **Documentation**: [Online Docs](https://unicolab.github.io/KerasFactory/)
- **Discord**: [![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.gg/6zf4VZFYV5)

---

<p align="center">
  <strong>Built with ❤️ for the Keras community by 🦄 UnicoLab.ai</strong>
</p>