# CNN - Convolutional Neural Network Library

A C++ convolutional neural network library supporting both CPU and GPU (OpenCL) execution. Delegates dense (fully-connected) layers to the [ANN](https://github.com/naschpitz/ANN) library.

## Features

- Conv2D, ReLU, Pool (Max & Avg), and Flatten layers
- NCHW tensor layout (channels-first)
- Configurable padding strategies (VALID, SAME, FULL) and strides
- CPU and GPU execution via OpenCL with multi-GPU training support
- Training, predict, and evaluation modes
- JSON-based configuration and model serialization
- He weight initialization for ReLU activations
- Factory pattern — callers use `Core<T>` without knowing the backend

## Dependencies

- Qt Core and Qt Concurrent
- [ANN](https://github.com/naschpitz/ANN) (dense layer implementation)
- [OpenCLWrapper](https://github.com/naschpitz/openCLWrapper) (for GPU support)
- [nlohmann/json](https://github.com/nlohmann/json)

## Building

```bash
git clone --recursive https://github.com/naschpitz/CNN.git
cd CNN
mkdir build && cd build
cmake ..
make
```

## Configuration File Format

### Train Mode

```json
{
  "mode": "train",
  "device": "cpu",
  "inputShape": { "c": 1, "h": 28, "w": 28 },
  "cnnLayersConfig": [
    { "type": "conv", "numFilters": 8, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
    { "type": "relu" },
    { "type": "pool", "poolType": "max", "poolH": 2, "poolW": 2, "strideY": 2, "strideX": 2 },
    { "type": "flatten" }
  ],
  "denseLayersConfig": [
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "trainingConfig": {
    "numEpochs": 10,
    "learningRate": 0.01,
    "numThreads": 0
  }
}
```

### Predict Mode

```json
{
  "mode": "predict",
  "device": "cpu",
  "inputShape": { "c": 1, "h": 28, "w": 28 },
  "cnnLayersConfig": [
    { "type": "conv", "numFilters": 8, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
    { "type": "relu" },
    { "type": "pool", "poolType": "max", "poolH": 2, "poolW": 2, "strideY": 2, "strideX": 2 },
    { "type": "flatten" }
  ],
  "denseLayersConfig": [
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "parameters": {
    "cnnFilters": [...],
    "cnnBiases": [...],
    "denseWeights": [...],
    "denseBiases": [...]
  }
}
```

### Test Mode

```json
{
  "mode": "test",
  "device": "cpu",
  "inputShape": { "c": 1, "h": 28, "w": 28 },
  "cnnLayersConfig": [...],
  "denseLayersConfig": [...],
  "parameters": {
    "cnnFilters": [...],
    "cnnBiases": [...],
    "denseWeights": [...],
    "denseBiases": [...]
  }
}
```

## Configuration Fields

| Field | Description | Required |
|-------|-------------|----------|
| `mode` | Operation mode: `train`, `predict`, or `test` | Optional (default: `predict`) |
| `device` | Execution device: `cpu` or `gpu` | Optional (default: `cpu`) |
| `inputShape` | Input tensor shape (`c`, `h`, `w`) | Required |
| `cnnLayersConfig` | Array of CNN layer definitions | Required |
| `denseLayersConfig` | Array of dense layer definitions (ANN) | Required |
| `trainingConfig` | Training hyperparameters | Required for `train` mode |
| `parameters` | Trained CNN and dense parameters | Required for `predict`/`test` modes |

## CNN Layer Types

| Type | Fields | Description |
|------|--------|-------------|
| `conv` | `numFilters`, `filterH`, `filterW`, `strideY`, `strideX`, `slidingStrategy` | 2D convolution with learnable filters |
| `relu` | — | Element-wise ReLU activation |
| `pool` | `poolType`, `poolH`, `poolW`, `strideY`, `strideX` | Spatial downsampling (max or avg) |
| `flatten` | — | Reshape 3D tensor to 1D vector for dense layers |

## Sliding Strategies

| Strategy | Padding | Effect |
|----------|---------|--------|
| `valid` | 0 | Output shrinks; no padding |
| `same` | ⌊K/2⌋ | Preserves spatial dimensions (stride=1) |
| `full` | K − 1 | Output expands; maximum padding |

## Dense Layer Activation Functions

| Function | Description |
|----------|-------------|
| `none` | No activation (use for input layer) |
| `relu` | Rectified Linear Unit |
| `sigmoid` | Sigmoid function |
| `tanh` | Hyperbolic tangent |

## Output File Formats

### Trained Model Output (from training)

After training, the model file contains the full architecture and learned parameters:

```json
{
  "inputShape": { "c": 1, "h": 28, "w": 28 },
  "cnnLayersConfig": [...],
  "denseLayersConfig": [...],
  "parameters": {
    "cnnFilters": [...],
    "cnnBiases": [...],
    "denseWeights": [[[...], ...], ...],
    "denseBiases": [[...], ...]
  },
  "trainingMetadata": {
    "startTime": "2025-02-20T10:30:00",
    "endTime": "2025-02-20T10:45:30",
    "durationSeconds": 930.5,
    "durationFormatted": "15m 30.5s",
    "numEpochs": 10,
    "learningRate": 0.01,
    "numSamples": 60000,
    "finalLoss": 0.0234
  }
}
```

### Predict Output

```json
{
  "predictMetadata": {
    "startTime": "2025-02-20T11:00:00",
    "endTime": "2025-02-20T11:00:00",
    "durationSeconds": 0.0012,
    "durationFormatted": "1.2ms"
  },
  "output": [0.01, 0.02, 0.95, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0]
}
```

### Test/Evaluation Output

```json
{
  "testMetadata": {
    "startTime": "2025-02-20T11:05:00",
    "endTime": "2025-02-20T11:05:45",
    "durationSeconds": 45.3,
    "durationFormatted": "45.3s"
  },
  "results": {
    "numSamples": 10000,
    "totalLoss": 234.5,
    "averageLoss": 0.02345
  }
}
```

## Input File Formats

### Samples File (for training/testing)

Input values are flat arrays of C × H × W elements, reshaped internally using `inputShape`:

```json
{
  "samples": [
    {
      "input": [0.0, 0.5, 1.0, 0.75, ...],
      "output": [1.0, 0.0, 0.0, ...]
    },
    {
      "input": [1.0, 0.25, 0.0, 0.5, ...],
      "output": [0.0, 1.0, 0.0, ...]
    }
  ]
}
```

### Input File (for predict)

```json
{
  "input": [0.0, 0.5, 1.0, 0.75, ...]
}
```

## Documentation

Comprehensive HTML documentation is available in the [`html/`](html/) directory. Open `html/index.html` in a browser to browse the API reference, architecture overview, layer documentation, mathematical foundations, and GPU implementation details.

## License

See [LICENSE.md](LICENSE.md) for details.
