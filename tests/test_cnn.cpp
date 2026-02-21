#include "CNN_Types.hpp"
#include "CNN_Conv2D.hpp"
#include "CNN_ReLU.hpp"
#include "CNN_Pool.hpp"
#include "CNN_Flatten.hpp"
#include "CNN_Core.hpp"
#include "CNN_CoreConfig.hpp"
#include "CNN_Sample.hpp"

#include <ANN_ActvFunc.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

static int testsPassed = 0;
static int testsFailed = 0;

#define CHECK(cond, msg) do { \
  if (!(cond)) { \
    std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    testsFailed++; \
  } else { \
    testsPassed++; \
  } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) CHECK(std::fabs((a) - (b)) < (tol), msg)

//===================================================================================================================//

void testTensor3D() {
  std::cout << "--- testTensor3D ---" << std::endl;

  CNN::Tensor3D<double> t({2, 3, 4});
  CHECK(t.shape.c == 2 && t.shape.h == 3 && t.shape.w == 4, "shape");
  CHECK(t.size() == 24, "size");
  CHECK(t.at(0, 0, 0) == 0.0, "zero-initialized");

  t.at(1, 2, 3) = 42.0;
  CHECK_NEAR(t.at(1, 2, 3), 42.0, 1e-9, "at accessor");

  CNN::Tensor3D<double> t2({1, 2, 2}, 5.0);
  CHECK_NEAR(t2.at(0, 1, 1), 5.0, 1e-9, "fill constructor");
}

//===================================================================================================================//

void testConv2DForward() {
  std::cout << "--- testConv2DForward ---" << std::endl;

  CNN::Tensor3D<double> input({1, 4, 4});
  for (ulong i = 0; i < 16; i++) input.data[i] = static_cast<double>(i + 1);

  CNN::ConvLayerConfig config{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};
  CNN::ConvParameters<double> params;
  params.numFilters = 1; params.inputC = 1; params.filterH = 3; params.filterW = 3;
  params.filters.assign(9, 1.0);
  params.biases.assign(1, 0.0);

  CNN::Tensor3D<double> out = CNN::Conv2D<double>::predict(input, config, params);
  CHECK(out.shape.c == 1 && out.shape.h == 2 && out.shape.w == 2, "conv2d output shape");
  CHECK_NEAR(out.at(0, 0, 0), 54.0, 1e-9, "conv2d top-left");
  CHECK_NEAR(out.at(0, 0, 1), 63.0, 1e-9, "conv2d top-right");
  CHECK_NEAR(out.at(0, 1, 0), 90.0, 1e-9, "conv2d bottom-left");
  CHECK_NEAR(out.at(0, 1, 1), 99.0, 1e-9, "conv2d bottom-right");
}

//===================================================================================================================//

void testConv2DBackprop() {
  std::cout << "--- testConv2DBackprop ---" << std::endl;

  CNN::Tensor3D<double> input({1, 4, 4});
  for (ulong i = 0; i < 16; i++) input.data[i] = static_cast<double>(i + 1);

  CNN::ConvLayerConfig config{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};
  CNN::ConvParameters<double> params;
  params.numFilters = 1; params.inputC = 1; params.filterH = 3; params.filterW = 3;
  params.filters.assign(9, 1.0);
  params.biases.assign(1, 0.0);

  CNN::Tensor3D<double> dOut({1, 2, 2}, 1.0);
  std::vector<double> dFilters, dBiases;
  CNN::Tensor3D<double> dInput = CNN::Conv2D<double>::backpropagate(dOut, input, config, params, dFilters, dBiases);

  CHECK(dInput.shape.c == 1 && dInput.shape.h == 4 && dInput.shape.w == 4, "backprop dInput shape");
  CHECK(dFilters.size() == 9, "backprop dFilters size");
  CHECK(dBiases.size() == 1, "backprop dBiases size");
  CHECK_NEAR(dBiases[0], 4.0, 1e-9, "backprop dBias value");
}

//===================================================================================================================//

void testReLU() {
  std::cout << "--- testReLU ---" << std::endl;

  CNN::Tensor3D<double> input({1, 2, 3});
  input.data = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

  CNN::Tensor3D<double> out = CNN::ReLU<double>::predict(input);
  CHECK_NEAR(out.data[0], 0.0, 1e-9, "relu neg -> 0");
  CHECK_NEAR(out.data[3], 1.0, 1e-9, "relu pos passthrough");
  CHECK_NEAR(out.data[5], 3.0, 1e-9, "relu pos passthrough");

  CNN::Tensor3D<double> dOut({1, 2, 3}, 1.0);
  CNN::Tensor3D<double> dIn = CNN::ReLU<double>::backpropagate(dOut, input);
  CHECK_NEAR(dIn.data[0], 0.0, 1e-9, "relu backprop blocks neg");
  CHECK_NEAR(dIn.data[3], 1.0, 1e-9, "relu backprop passes pos");
}

//===================================================================================================================//

void testMaxPool() {
  std::cout << "--- testMaxPool ---" << std::endl;

  CNN::Tensor3D<double> input({1, 4, 4});
  for (ulong i = 0; i < 16; i++) input.data[i] = static_cast<double>(i + 1);

  CNN::PoolLayerConfig config{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};
  std::vector<ulong> maxIndices;
  CNN::Tensor3D<double> out = CNN::Pool<double>::predict(input, config, maxIndices);

  CHECK(out.shape.c == 1 && out.shape.h == 2 && out.shape.w == 2, "maxpool shape");
  CHECK_NEAR(out.at(0, 0, 0), 6.0, 1e-9, "maxpool top-left");
  CHECK_NEAR(out.at(0, 0, 1), 8.0, 1e-9, "maxpool top-right");
  CHECK_NEAR(out.at(0, 1, 0), 14.0, 1e-9, "maxpool bottom-left");
  CHECK_NEAR(out.at(0, 1, 1), 16.0, 1e-9, "maxpool bottom-right");

  CNN::Tensor3D<double> dOut({1, 2, 2}, 1.0);
  CNN::Tensor3D<double> dIn = CNN::Pool<double>::backpropagate(dOut, input.shape, config, maxIndices);
  CHECK(dIn.shape.h == 4 && dIn.shape.w == 4, "maxpool backprop shape");
  CHECK_NEAR(dIn.at(0, 1, 1), 1.0, 1e-9, "maxpool backprop max pos");
  CHECK_NEAR(dIn.at(0, 0, 0), 0.0, 1e-9, "maxpool backprop non-max");
}

//===================================================================================================================//

void testFlatten() {
  std::cout << "--- testFlatten ---" << std::endl;

  CNN::Tensor3D<double> input({2, 3, 4});
  for (ulong i = 0; i < 24; i++) input.data[i] = static_cast<double>(i);

  CNN::Tensor1D<double> flat = CNN::Flatten<double>::predict(input);
  CHECK(flat.size() == 24, "flatten size");
  CHECK_NEAR(flat[0], 0.0, 1e-9, "flatten first");
  CHECK_NEAR(flat[23], 23.0, 1e-9, "flatten last");

  CNN::Tensor3D<double> back = CNN::Flatten<double>::backpropagate(flat, input.shape);
  CHECK(back.shape.c == 2 && back.shape.h == 3 && back.shape.w == 4, "unflatten shape");
  CHECK_NEAR(back.at(1, 2, 3), 23.0, 1e-9, "unflatten value");
}



//===================================================================================================================//

void testEndToEnd() {
  std::cout << "--- testEndToEnd (train + predict) ---" << std::endl;

  // 1x5x5 → Conv(1 filter 3x3 valid) → 1x3x3 → ReLU → Flatten(9) → Dense(1, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.verbose = false;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  // Supply pre-initialized conv parameters to avoid dead-ReLU from random init.
  // With all-ones input, conv output = sum(filters) + bias. If sum <= 0, ReLU zeros
  // everything out and gradients are blocked. Using positive filter values ensures
  // the network starts in a learning-friendly state.
  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1);  // All positive → sum = 0.9 > 0
  initConv.biases.assign(1, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.progressReports = 0;

  auto core = CNN::Core<double>::makeCore(config);
  CHECK(core != nullptr, "core creation");

  // "bright" (all 1s) → 1, "dark" (all 0s) → 0
  CNN::Samples<double> samples(2);
  samples[0].input = CNN::Tensor3D<double>({1, 5, 5}, 1.0);
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  core->train(samples);

  CNN::Output<double> pred0 = core->predict(samples[0].input);
  CNN::Output<double> pred1 = core->predict(samples[1].input);

  CHECK(pred0.size() == 1, "predict output size 0");
  CHECK(pred1.size() == 1, "predict output size 1");
  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(pred0[0] > pred1[0], "bright > dark after training");

  // Test method
  CNN::TestResult<double> result = core->test(samples);
  CHECK(result.numSamples == 2, "test numSamples");
  CHECK(result.averageLoss >= 0.0, "test avgLoss non-negative");
  std::cout << "  test avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

int main() {
  std::cout << "=== CNN Unit Tests ===" << std::endl;

  testTensor3D();
  testConv2DForward();
  testConv2DBackprop();
  testReLU();
  testMaxPool();
  testFlatten();
  testEndToEnd();

  std::cout << std::endl;
  std::cout << "=== Results: " << testsPassed << " passed, " << testsFailed << " failed ===" << std::endl;

  return (testsFailed > 0) ? 1 : 0;
}