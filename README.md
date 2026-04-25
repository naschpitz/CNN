# CNN - Convolutional Neural Network Library

A C++ convolutional neural network library supporting both CPU and GPU (OpenCL) execution. Delegates dense (fully-connected) layers to the [ANN](https://github.com/naschpitz/ANN) library.

## Features

- Conv2D, ReLU, Pool (Max & Avg), and Flatten layers
- NCHW tensor layout (channels-first)
- Configurable padding strategies (VALID, SAME, FULL) and strides
- CPU and GPU execution via OpenCL with multi-GPU training support
- Training, predict, and evaluation modes
- He weight initialization for ReLU activations
- Factory pattern — callers use `Core<T>` without knowing the backend
- Pure library — accessible entirely via C++ function calls

## Dependencies

- Qt Core and Qt Concurrent
- [ANN](https://github.com/naschpitz/ANN) (dense layer implementation)
- [OpenCLWrapper](https://github.com/naschpitz/openCLWrapper) (for GPU support)

## Building

```bash
git clone --recursive https://github.com/naschpitz/CNN.git
cd CNN
mkdir build && cd build
cmake ..
make
```

## API Overview

CNN is a pure library — all interaction is through C++ types. There is no built-in file I/O;
the calling application (e.g. [NN-CLI](https://github.com/naschpitz/NN-CLI)) handles serialization.

```cpp
#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>
#include <CNN_Sample.hpp>
#include <CNN_SampleProvider.hpp>

// Configure
CNN::CoreConfig<float> config;
config.modeType   = CNN::ModeType::TRAIN;
config.deviceType = CNN::DeviceType::CPU;
config.numThreads = 0;             // 0 = use all available CPU cores
config.numGPUs    = 0;             // 0 = use all available GPUs (GPU mode)
config.inputShape = {1, 28, 28};   // C, H, W

config.layersConfig.cnnLayers = {
  {CNN::LayerType::CONV, CNN::ConvLayerConfig{8, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}},
  {CNN::LayerType::RELU, CNN::ReLULayerConfig{}},
  {CNN::LayerType::POOL, CNN::PoolLayerConfig{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2}},
  {CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}}
};
config.layersConfig.denseLayers = {
  {128, ANN::ActvFuncType::RELU},
  { 10, ANN::ActvFuncType::SIGMOID}
};
config.trainingConfig.numEpochs    = 10;
config.trainingConfig.batchSize    = 64;
config.trainingConfig.learningRate = 0.01f;

config.testConfig.batchSize = 128;   // batch size for test evaluation (default: 64)

// Create, train, test, and query
auto core = CNN::Core<float>::makeCore(config);
core->train(samples.size(), CNN::makeSampleProvider(samples));

auto testResult = core->test(testSamples.size(), CNN::makeSampleProvider(testSamples));
// testResult.accuracy, testResult.averageLoss, testResult.numCorrect, ...

const auto& params = core->getParameters();   // conv + dense params
const auto& meta   = core->getTrainingMetadata();
```

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

## Documentation

Comprehensive HTML documentation is available in the [`docs/`](docs/) directory. Open `docs/index.html` in a browser to browse the API reference, architecture overview, layer documentation, mathematical foundations, and GPU implementation details.

## License

See [LICENSE.md](LICENSE.md) for details.
