#include "test_helpers.hpp"
#include "CNN_Device.hpp"
#include "CNN_Mode.hpp"
#include "CNN_PoolType.hpp"
#include "CNN_CostFunctionConfig.hpp"

//===================================================================================================================//

static void testTensor3D()
{
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

static void testReLU()
{
  std::cout << "--- testReLU ---" << std::endl;

  CNN::Tensor3D<double> input({1, 2, 3});
  input.data = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

  CNN::Tensor3D<double> out = CNN::ReLU<double>::propagate(input);
  CHECK_NEAR(out.data[0], 0.0, 1e-9, "relu neg -> 0");
  CHECK_NEAR(out.data[3], 1.0, 1e-9, "relu pos passthrough");
  CHECK_NEAR(out.data[5], 3.0, 1e-9, "relu pos passthrough");

  CNN::Tensor3D<double> dOut({1, 2, 3}, 1.0);
  CNN::Tensor3D<double> dIn = CNN::ReLU<double>::backpropagate(dOut, input);
  CHECK_NEAR(dIn.data[0], 0.0, 1e-9, "relu backprop blocks neg");
  CHECK_NEAR(dIn.data[3], 1.0, 1e-9, "relu backprop passes pos");
}

//===================================================================================================================//

static void testMaxPool()
{
  std::cout << "--- testMaxPool ---" << std::endl;

  CNN::Tensor3D<double> input({1, 4, 4});

  for (ulong i = 0; i < 16; i++)
    input.data[i] = static_cast<double>(i + 1);

  CNN::PoolLayerConfig config{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};
  std::vector<ulong> maxIndices;
  CNN::Tensor3D<double> out = CNN::Pool<double>::propagate(input, config, maxIndices);

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

static void testAvgPool()
{
  std::cout << "--- testAvgPool ---" << std::endl;

  CNN::Tensor3D<double> input({1, 4, 4});

  for (ulong i = 0; i < 16; i++)
    input.data[i] = static_cast<double>(i + 1);

  CNN::PoolLayerConfig config{CNN::PoolTypeEnum::AVG, 2, 2, 2, 2};
  std::vector<ulong> maxIndices;
  CNN::Tensor3D<double> out = CNN::Pool<double>::propagate(input, config, maxIndices);

  CHECK(out.shape.c == 1 && out.shape.h == 2 && out.shape.w == 2, "avgpool shape");
  CHECK(maxIndices.empty(), "avgpool no maxIndices");
  // Top-left: (1+2+5+6)/4 = 14/4 = 3.5
  CHECK_NEAR(out.at(0, 0, 0), 3.5, 1e-9, "avgpool top-left");
  // Bottom-right: (11+12+15+16)/4 = 54/4 = 13.5
  CHECK_NEAR(out.at(0, 1, 1), 13.5, 1e-9, "avgpool bottom-right");

  // Backprop: gradient distributed evenly
  CNN::Tensor3D<double> dOut({1, 2, 2}, 4.0); // gradient = 4.0
  CNN::Tensor3D<double> dIn = CNN::Pool<double>::backpropagate(dOut, input.shape, config, maxIndices);
  // Each element gets 4.0 / (2*2) = 1.0
  CHECK_NEAR(dIn.at(0, 0, 0), 1.0, 1e-9, "avgpool backprop");
  CHECK_NEAR(dIn.at(0, 1, 1), 1.0, 1e-9, "avgpool backprop center");
}

//===================================================================================================================//

static void testPoolNonSquare()
{
  std::cout << "--- testPoolNonSquare ---" << std::endl;

  // 1x4x6 input, pool 2x3 stride 2x3 → 1x2x2
  CNN::Tensor3D<double> input({1, 4, 6});

  for (ulong i = 0; i < 24; i++)
    input.data[i] = static_cast<double>(i + 1);
  // Row 0: 1  2  3  4  5  6
  // Row 1: 7  8  9  10 11 12
  // Row 2: 13 14 15 16 17 18
  // Row 3: 19 20 21 22 23 24

  CNN::PoolLayerConfig config{CNN::PoolTypeEnum::MAX, 2, 3, 2, 3};
  std::vector<ulong> maxIndices;
  CNN::Tensor3D<double> out = CNN::Pool<double>::propagate(input, config, maxIndices);

  CHECK(out.shape.c == 1 && out.shape.h == 2 && out.shape.w == 2, "non-square pool shape");
  // (oh=0,ow=0): rows 0-1, cols 0-2 → max(1,2,3,7,8,9) = 9
  CHECK_NEAR(out.at(0, 0, 0), 9.0, 1e-9, "non-square pool [0,0]");
  // (oh=0,ow=1): rows 0-1, cols 3-5 → max(4,5,6,10,11,12) = 12
  CHECK_NEAR(out.at(0, 0, 1), 12.0, 1e-9, "non-square pool [0,1]");
  // (oh=1,ow=1): rows 2-3, cols 3-5 → max(16,17,18,22,23,24) = 24
  CHECK_NEAR(out.at(0, 1, 1), 24.0, 1e-9, "non-square pool [1,1]");
}

//===================================================================================================================//

static void testFlatten()
{
  std::cout << "--- testFlatten ---" << std::endl;

  CNN::Tensor3D<double> input({2, 3, 4});

  for (ulong i = 0; i < 24; i++)
    input.data[i] = static_cast<double>(i);

  CNN::Tensor1D<double> flat = CNN::Flatten<double>::propagate(input);
  CHECK(flat.size() == 24, "flatten size");
  CHECK_NEAR(flat[0], 0.0, 1e-9, "flatten first");
  CHECK_NEAR(flat[23], 23.0, 1e-9, "flatten last");

  CNN::Tensor3D<double> back = CNN::Flatten<double>::backpropagate(flat, input.shape);
  CHECK(back.shape.c == 2 && back.shape.h == 3 && back.shape.w == 4, "unflatten shape");
  CHECK_NEAR(back.at(1, 2, 3), 23.0, 1e-9, "unflatten value");
}

//===================================================================================================================//

static void testInstanceNormInference()
{
  std::cout << "--- testInstanceNormInference ---" << std::endl;

  // 2 channels, 2x2 spatial
  CNN::Shape3D shape{2, 2, 2};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0, // channel 0
                5.0, 6.0, 7.0, 8.0}; // channel 1

  CNN::InstanceNormParameters<double> params;
  params.numChannels = 2;
  params.gamma = {2.0, 0.5};
  params.beta = {1.0, -1.0};
  params.runningMean = {2.5, 6.5};
  params.runningVar = {1.25, 1.25};

  CNN::InstanceNormLayerConfig config;
  config.epsilon = 0.0; // zero eps for exact math

  CNN::Tensor3D<double> out = CNN::InstanceNorm<double>::propagate(input, shape, params, config);

  CHECK(out.shape.c == 2 && out.shape.h == 2 && out.shape.w == 2, "instancenorm inference shape");

  // channel 0: invStd = 1/sqrt(1.25) ≈ 0.894427
  // out[0] = 2.0 * (1.0 - 2.5) * invStd + 1.0 = 2.0 * (-1.5) * 0.894427 + 1.0 = -1.68328...
  double invStd0 = 1.0 / std::sqrt(1.25);

  CHECK_NEAR(out.data[0], 2.0 * (1.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [0]");
  CHECK_NEAR(out.data[1], 2.0 * (2.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [1]");
  CHECK_NEAR(out.data[2], 2.0 * (3.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [2]");
  CHECK_NEAR(out.data[3], 2.0 * (4.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [3]");

  // channel 1: same invStd
  CHECK_NEAR(out.data[4], 0.5 * (5.0 - 6.5) * invStd0 - 1.0, 1e-9, "instancenorm infer ch1 [0]");
  CHECK_NEAR(out.data[7], 0.5 * (8.0 - 6.5) * invStd0 - 1.0, 1e-9, "instancenorm infer ch1 [3]");

  // Running stats should not change during inference
  CHECK_NEAR(params.runningMean[0], 2.5, 1e-9, "instancenorm infer runningMean unchanged");
  CHECK_NEAR(params.runningVar[0], 1.25, 1e-9, "instancenorm infer runningVar unchanged");
}

//===================================================================================================================//

static void testInstanceNormTraining()
{
  std::cout << "--- testInstanceNormTraining ---" << std::endl;

  // 2 channels, 2x2 spatial
  CNN::Shape3D shape{2, 2, 2};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0, // channel 0
                5.0, 6.0, 7.0, 8.0}; // channel 1

  CNN::InstanceNormParameters<double> params;
  params.numChannels = 2;
  params.gamma = {1.0, 1.0};
  params.beta = {0.0, 0.0};
  params.runningMean = {0.0, 0.0};
  params.runningVar = {1.0, 1.0};

  CNN::InstanceNormLayerConfig config;
  config.epsilon = 0.0;
  config.momentum = 0.1;

  std::vector<double> batchMean, batchVar;
  CNN::Tensor3D<double> xNorm;

  CNN::Tensor3D<double> out =
    CNN::InstanceNorm<double>::propagate(input, shape, params, config, true, &batchMean, &batchVar, &xNorm);

  CHECK(out.shape.c == 2 && out.shape.h == 2 && out.shape.w == 2, "instancenorm train shape");

  // channel 0: mean = (1+2+3+4)/4 = 2.5, var = ((−1.5)²+(−0.5)²+(0.5)²+(1.5)²)/4 = 1.25
  CHECK_NEAR(batchMean[0], 2.5, 1e-9, "instancenorm train mean ch0");
  CHECK_NEAR(batchVar[0], 1.25, 1e-9, "instancenorm train var ch0");

  // channel 1: mean = (5+6+7+8)/4 = 6.5, var = 1.25
  CHECK_NEAR(batchMean[1], 6.5, 1e-9, "instancenorm train mean ch1");
  CHECK_NEAR(batchVar[1], 1.25, 1e-9, "instancenorm train var ch1");

  // With gamma=1, beta=0: output = xNormalized
  double invStd = 1.0 / std::sqrt(1.25);
  CHECK_NEAR(out.data[0], (1.0 - 2.5) * invStd, 1e-9, "instancenorm train out ch0 [0]");
  CHECK_NEAR(out.data[3], (4.0 - 2.5) * invStd, 1e-9, "instancenorm train out ch0 [3]");

  // xNormalized should match output when gamma=1, beta=0
  CHECK_NEAR(xNorm.data[0], out.data[0], 1e-9, "instancenorm train xNorm matches out");
  CHECK_NEAR(xNorm.data[7], out.data[7], 1e-9, "instancenorm train xNorm matches out ch1");

  // Running stats updated: runningMean = 0.9*0.0 + 0.1*2.5 = 0.25
  CHECK_NEAR(params.runningMean[0], 0.25, 1e-7, "instancenorm train runningMean updated ch0");
  CHECK_NEAR(params.runningMean[1], 0.65, 1e-7, "instancenorm train runningMean updated ch1");
  // runningVar = 0.9*1.0 + 0.1*1.25 = 1.025
  CHECK_NEAR(params.runningVar[0], 1.025, 1e-7, "instancenorm train runningVar updated ch0");
}

//===================================================================================================================//

static void testInstanceNormBackpropagate()
{
  std::cout << "--- testInstanceNormBackpropagate ---" << std::endl;

  // 1 channel, 1x4 spatial for simple math
  CNN::Shape3D shape{1, 1, 4};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0};

  CNN::InstanceNormParameters<double> params;
  params.numChannels = 1;
  params.gamma = {2.0};
  params.beta = {0.5};
  params.runningMean = {0.0};
  params.runningVar = {1.0};

  CNN::InstanceNormLayerConfig config;
  config.epsilon = 0.0;
  config.momentum = 0.1;

  // Forward pass (training) to get intermediates
  std::vector<double> batchMean, batchVar;
  CNN::Tensor3D<double> xNorm;
  CNN::InstanceNorm<double>::propagate(input, shape, params, config, true, &batchMean, &batchVar, &xNorm);

  // mean=2.5, var=1.25
  CHECK_NEAR(batchMean[0], 2.5, 1e-9, "backprop setup mean");
  CHECK_NEAR(batchVar[0], 1.25, 1e-9, "backprop setup var");

  // Upstream gradient: all ones
  CNN::Tensor3D<double> dOutput(shape, 1.0);

  std::vector<double> dGamma, dBeta;
  CNN::Tensor3D<double> dInput =
    CNN::InstanceNorm<double>::backpropagate(dOutput, shape, params, config, batchMean, batchVar, xNorm, dGamma, dBeta);

  // dBeta = sum(dOutput) = 4.0
  CHECK_NEAR(dBeta[0], 4.0, 1e-9, "instancenorm backprop dBeta");

  // dGamma = sum(dOutput * xNorm) = sum(xNorm) = sum of normalized values = 0 (by construction)
  CHECK_NEAR(dGamma[0], 0.0, 1e-9, "instancenorm backprop dGamma");

  // dInput shape should match
  CHECK(dInput.shape.c == 1 && dInput.shape.h == 1 && dInput.shape.w == 4, "instancenorm backprop shape");

  // With uniform dOutput=1 and dGamma=0: dInput = gamma*invStd/N * (N*1 - 4 - xNorm*0) = gamma*invStd*(1 - 1) = 0
  // Actually: dInput_i = (gamma / (N*invStd_inv)) * (N*dOut_i - dBeta - xNorm_i*dGamma)
  // = (2.0 / (4 * sqrt(1.25))) * (4*1 - 4 - xNorm_i*0) = 0
  for (ulong i = 0; i < 4; i++)
    CHECK_NEAR(dInput.data[i], 0.0, 1e-9, "instancenorm backprop dInput uniform=0");
}

//===================================================================================================================//

static void testInstanceNormBackpropGradient()
{
  std::cout << "--- testInstanceNormBackpropGradient ---" << std::endl;

  // Non-uniform gradient to test actual gradient flow
  CNN::Shape3D shape{1, 1, 4};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0};

  CNN::InstanceNormParameters<double> params;
  params.numChannels = 1;
  params.gamma = {1.0};
  params.beta = {0.0};
  params.runningMean = {0.0};
  params.runningVar = {1.0};

  CNN::InstanceNormLayerConfig config;
  config.epsilon = 1e-7;
  config.momentum = 0.1;

  // Forward pass
  std::vector<double> batchMean, batchVar;
  CNN::Tensor3D<double> xNorm;
  CNN::InstanceNorm<double>::propagate(input, shape, params, config, true, &batchMean, &batchVar, &xNorm);

  // Non-uniform upstream gradient
  CNN::Tensor3D<double> dOutput(shape);
  dOutput.data = {0.1, 0.2, 0.3, 0.4};

  std::vector<double> dGamma, dBeta;
  CNN::Tensor3D<double> dInput =
    CNN::InstanceNorm<double>::backpropagate(dOutput, shape, params, config, batchMean, batchVar, xNorm, dGamma, dBeta);

  // dBeta = sum(dOutput) = 1.0
  CHECK_NEAR(dBeta[0], 1.0, 1e-9, "grad dBeta");

  // Numerical gradient check: perturb each input by eps and compute finite difference
  double eps = 1e-5;

  for (ulong i = 0; i < 4; i++) {
    // Forward with input[i] + eps
    CNN::Tensor3D<double> inputPlus(shape);
    inputPlus.data = input.data;
    inputPlus.data[i] += eps;

    CNN::InstanceNormParameters<double> pPlus = params;
    pPlus.runningMean = {0.0};
    pPlus.runningVar = {1.0};
    std::vector<double> bmP, bvP;
    CNN::Tensor3D<double> xnP;
    CNN::Tensor3D<double> outPlus =
      CNN::InstanceNorm<double>::propagate(inputPlus, shape, pPlus, config, true, &bmP, &bvP, &xnP);

    // Forward with input[i] - eps
    CNN::Tensor3D<double> inputMinus(shape);
    inputMinus.data = input.data;
    inputMinus.data[i] -= eps;

    CNN::InstanceNormParameters<double> pMinus = params;
    pMinus.runningMean = {0.0};
    pMinus.runningVar = {1.0};
    std::vector<double> bmM, bvM;
    CNN::Tensor3D<double> xnM;
    CNN::Tensor3D<double> outMinus =
      CNN::InstanceNorm<double>::propagate(inputMinus, shape, pMinus, config, true, &bmM, &bvM, &xnM);

    // Numerical gradient = sum_j dOutput[j] * (outPlus[j] - outMinus[j]) / (2*eps)
    double numGrad = 0.0;

    for (ulong j = 0; j < 4; j++)
      numGrad += dOutput.data[j] * (outPlus.data[j] - outMinus.data[j]) / (2.0 * eps);

    CHECK_NEAR(dInput.data[i], numGrad, 1e-4, "instancenorm numerical gradient check");
  }
}

//===================================================================================================================//

static void testInstanceNormOutputShape()
{
  std::cout << "--- testInstanceNormOutputShape ---" << std::endl;

  // InstanceNorm should not change shape
  CNN::Shape3D shape{3, 8, 8};
  CNN::Tensor3D<double> input(shape, 1.0);

  CNN::InstanceNormParameters<double> params;
  params.numChannels = 3;
  params.gamma = {1.0, 1.0, 1.0};
  params.beta = {0.0, 0.0, 0.0};
  params.runningMean = {0.0, 0.0, 0.0};
  params.runningVar = {1.0, 1.0, 1.0};

  CNN::InstanceNormLayerConfig config;

  CNN::Tensor3D<double> out = CNN::InstanceNorm<double>::propagate(input, shape, params, config);
  CHECK(out.shape.c == 3, "instancenorm preserves channels");
  CHECK(out.shape.h == 8, "instancenorm preserves height");
  CHECK(out.shape.w == 8, "instancenorm preserves width");
}

//===================================================================================================================//

static void testInstanceNormValidateShapes()
{
  std::cout << "--- testInstanceNormValidateShapes ---" << std::endl;

  CNN::LayersConfig lc;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig bn;
  bn.type = CNN::LayerType::INSTANCENORM;
  bn.config = CNN::InstanceNormLayerConfig{};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flatten;
  flatten.type = CNN::LayerType::FLATTEN;
  flatten.config = CNN::FlattenLayerConfig{};

  lc.cnnLayers = {conv1, bn, relu1, flatten};

  // 1x8x8 → Conv(4,3x3,valid) → 4x6x6 → BN → 4x6x6 → ReLU → 4x6x6
  CNN::Shape3D outShape = lc.validateShapes({1, 8, 8});
  CHECK(outShape.c == 4 && outShape.h == 6 && outShape.w == 6, "validateShapes with instancenorm");
}

//===================================================================================================================//

static void testSlidingStrategy()
{
  std::cout << "--- testSlidingStrategy ---" << std::endl;

  CHECK(CNN::SlidingStrategy::computePadding(3, CNN::SlidingStrategyType::VALID) == 0, "valid pad=0");
  CHECK(CNN::SlidingStrategy::computePadding(5, CNN::SlidingStrategyType::VALID) == 0, "valid pad=0 k5");
  CHECK(CNN::SlidingStrategy::computePadding(3, CNN::SlidingStrategyType::FULL) == 2, "full pad k3");
  CHECK(CNN::SlidingStrategy::computePadding(5, CNN::SlidingStrategyType::FULL) == 4, "full pad k5");
  CHECK(CNN::SlidingStrategy::computePadding(3, CNN::SlidingStrategyType::SAME) == 1, "same pad k3");
  CHECK(CNN::SlidingStrategy::computePadding(5, CNN::SlidingStrategyType::SAME) == 2, "same pad k5");

  CHECK(CNN::SlidingStrategy::nameToType("valid") == CNN::SlidingStrategyType::VALID, "nameToType valid");
  CHECK(CNN::SlidingStrategy::nameToType("same") == CNN::SlidingStrategyType::SAME, "nameToType same");
  CHECK(CNN::SlidingStrategy::nameToType("full") == CNN::SlidingStrategyType::FULL, "nameToType full");

  CHECK(CNN::SlidingStrategy::typeToName(CNN::SlidingStrategyType::VALID) == "valid", "typeToName valid");

  CHECK_THROWS(CNN::SlidingStrategy::nameToType("bogus"), "nameToType unknown throws");
  CHECK_THROWS(CNN::SlidingStrategy::nameToType(""), "nameToType empty throws");
}

//===================================================================================================================//

static void testDeviceNameToType()
{
  std::cout << "--- testDeviceNameToType ---" << std::endl;

  CHECK(CNN::Device::nameToType("cpu") == CNN::DeviceType::CPU, "cpu → CPU");
  CHECK(CNN::Device::nameToType("gpu") == CNN::DeviceType::GPU, "gpu → GPU");

  CHECK_THROWS(CNN::Device::nameToType("nonexistent"), "nonexistent throws");
  CHECK_THROWS(CNN::Device::nameToType(""), "empty throws");

  CHECK(CNN::Device::typeToName(CNN::DeviceType::CPU) == "cpu", "CPU → cpu");
  CHECK(CNN::Device::typeToName(CNN::DeviceType::GPU) == "gpu", "GPU → gpu");
}

//===================================================================================================================//

static void testModeNameToType()
{
  std::cout << "--- testModeNameToType ---" << std::endl;

  CHECK(CNN::Mode::nameToType("train") == CNN::ModeType::TRAIN, "train → TRAIN");
  CHECK(CNN::Mode::nameToType("predict") == CNN::ModeType::PREDICT, "predict → PREDICT");
  CHECK(CNN::Mode::nameToType("test") == CNN::ModeType::TEST, "test → TEST");

  CHECK_THROWS(CNN::Mode::nameToType("nonexistent"), "nonexistent throws");
  CHECK_THROWS(CNN::Mode::nameToType(""), "empty throws");

  CHECK(CNN::Mode::typeToName(CNN::ModeType::TRAIN) == "train", "TRAIN → train");
  CHECK(CNN::Mode::typeToName(CNN::ModeType::PREDICT) == "predict", "PREDICT → predict");
  CHECK(CNN::Mode::typeToName(CNN::ModeType::TEST) == "test", "TEST → test");
}

//===================================================================================================================//

static void testPoolTypeNameToType()
{
  std::cout << "--- testPoolTypeNameToType ---" << std::endl;

  CHECK(CNN::PoolType::nameToType("max") == CNN::PoolTypeEnum::MAX, "max → MAX");
  CHECK(CNN::PoolType::nameToType("avg") == CNN::PoolTypeEnum::AVG, "avg → AVG");

  CHECK_THROWS(CNN::PoolType::nameToType("nonexistent"), "nonexistent throws");
  CHECK_THROWS(CNN::PoolType::nameToType(""), "empty throws");

  CHECK(CNN::PoolType::typeToName(CNN::PoolTypeEnum::MAX) == "max", "MAX → max");
  CHECK(CNN::PoolType::typeToName(CNN::PoolTypeEnum::AVG) == "avg", "AVG → avg");
}

//===================================================================================================================//

static void testCostFunctionNameToType()
{
  std::cout << "--- testCostFunctionNameToType ---" << std::endl;

  CHECK(CNN::CostFunction::nameToType("squaredDifference") == CNN::CostFunctionType::SQUARED_DIFFERENCE,
        "squaredDifference → SQUARED_DIFFERENCE");
  CHECK(CNN::CostFunction::nameToType("weightedSquaredDifference") ==
          CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE,
        "weightedSquaredDifference → WEIGHTED_SQUARED_DIFFERENCE");

  CHECK_THROWS(CNN::CostFunction::nameToType("nonexistent"), "nonexistent throws");
  CHECK_THROWS(CNN::CostFunction::nameToType(""), "empty throws");

  CHECK(CNN::CostFunction::typeToName(CNN::CostFunctionType::SQUARED_DIFFERENCE) == "squaredDifference",
        "SQUARED_DIFFERENCE → squaredDifference");
  CHECK(CNN::CostFunction::typeToName(CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE) ==
          "weightedSquaredDifference",
        "WEIGHTED_SQUARED_DIFFERENCE → weightedSquaredDifference");

  // Cross-entropy
  CHECK(CNN::CostFunction::nameToType("crossEntropy") == CNN::CostFunctionType::CROSS_ENTROPY,
        "crossEntropy → CROSS_ENTROPY");
  CHECK(CNN::CostFunction::typeToName(CNN::CostFunctionType::CROSS_ENTROPY) == "crossEntropy",
        "CROSS_ENTROPY → crossEntropy");
}

//===================================================================================================================//

static void testValidateShapes()
{
  std::cout << "--- testValidateShapes ---" << std::endl;

  CNN::LayersConfig lc;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig pool1;
  pool1.type = CNN::LayerType::POOL;
  pool1.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};

  CNN::CNNLayerConfig flatten;
  flatten.type = CNN::LayerType::FLATTEN;
  flatten.config = CNN::FlattenLayerConfig{};

  lc.cnnLayers = {conv1, relu1, pool1, flatten};

  // 1x8x8 → Conv(4,3x3,valid) → 4x6x6 → ReLU → 4x6x6 → Pool(2x2,s2) → 4x3x3
  CNN::Shape3D outShape = lc.validateShapes({1, 8, 8});
  CHECK(outShape.c == 4 && outShape.h == 3 && outShape.w == 3, "validateShapes output");

  // Test with invalid config (filter bigger than input)
  CNN::LayersConfig lc2;
  CNN::CNNLayerConfig badConv;
  badConv.type = CNN::LayerType::CONV;
  badConv.config = CNN::ConvLayerConfig{1, 10, 10, 1, 1, CNN::SlidingStrategyType::VALID};
  lc2.cnnLayers = {badConv};

  bool caught = false;

  try {
    lc2.validateShapes({1, 3, 3});
  } catch (const std::runtime_error&) {
    caught = true;
  }

  CHECK(caught, "validateShapes throws for oversized filter");
}

//===================================================================================================================//

void runLayerTests()
{
  testTensor3D();
  testReLU();
  testMaxPool();
  testAvgPool();
  testPoolNonSquare();
  testFlatten();
  testInstanceNormInference();
  testInstanceNormTraining();
  testInstanceNormBackpropagate();
  testInstanceNormBackpropGradient();
  testInstanceNormOutputShape();
  testInstanceNormValidateShapes();
  testSlidingStrategy();
  testDeviceNameToType();
  testModeNameToType();
  testPoolTypeNameToType();
  testCostFunctionNameToType();
  testValidateShapes();
}