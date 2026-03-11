#include "test_helpers.hpp"
#include "CNN_Device.hpp"
#include "CNN_Mode.hpp"
#include "CNN_PoolType.hpp"
#include "CNN_CostFunctionConfig.hpp"
#include "CNN_BatchNorm.hpp"

static void testInstanceNormOutputShape()
{
  std::cout << "--- testInstanceNormOutputShape ---" << std::endl;

  // InstanceNorm should not change shape
  CNN::Shape3D shape{3, 8, 8};
  CNN::Tensor3D<double> input(shape, 1.0);

  CNN::NormParameters<double> params;
  params.numChannels = 3;
  params.gamma = {1.0, 1.0, 1.0};
  params.beta = {0.0, 0.0, 0.0};
  params.runningMean = {0.0, 0.0, 0.0};
  params.runningVar = {1.0, 1.0, 1.0};

  CNN::NormLayerConfig config;

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
  bn.config = CNN::NormLayerConfig{};

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

static void testBatchNormInference()
{
  std::cout << "--- testBatchNormInference ---" << std::endl;

  // 2 channels, 2x2 spatial, 2 samples
  CNN::Shape3D shape{2, 2, 2};

  CNN::Tensor3D<double> s0(shape);
  s0.data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

  CNN::Tensor3D<double> s1(shape);
  s1.data = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  std::vector<CNN::Tensor3D<double>*> batch = {&s0, &s1};

  CNN::NormParameters<double> params;
  params.numChannels = 2;
  params.gamma = {2.0, 0.5};
  params.beta = {1.0, -1.0};
  params.runningMean = {2.5, 6.5};
  params.runningVar = {1.25, 1.25};

  CNN::NormLayerConfig config;
  config.epsilon = 0.0;

  CNN::BatchNorm<double>::propagate(batch, shape, params, config, false);

  // Inference uses running stats, same as InstanceNorm inference
  double invStd0 = 1.0 / std::sqrt(1.25);
  CHECK_NEAR(s0.data[0], 2.0 * (1.0 - 2.5) * invStd0 + 1.0, 1e-9, "batchnorm infer s0 ch0 [0]");
  CHECK_NEAR(s1.data[0], 2.0 * (2.0 - 2.5) * invStd0 + 1.0, 1e-9, "batchnorm infer s1 ch0 [0]");
}

//===================================================================================================================//

static void testBatchNormTraining()
{
  std::cout << "--- testBatchNormTraining ---" << std::endl;

  // 1 channel, 1x2 spatial, 2 samples
  CNN::Shape3D shape{1, 1, 2};

  CNN::Tensor3D<double> s0(shape);
  s0.data = {1.0, 3.0};

  CNN::Tensor3D<double> s1(shape);
  s1.data = {5.0, 7.0};

  std::vector<CNN::Tensor3D<double>*> batch = {&s0, &s1};

  CNN::NormParameters<double> params;
  params.numChannels = 1;
  params.gamma = {1.0};
  params.beta = {0.0};
  params.runningMean = {0.0};
  params.runningVar = {1.0};

  CNN::NormLayerConfig config;
  config.epsilon = 0.0;
  config.momentum = 0.1;

  std::vector<CNN::Tensor3D<double>> xNorm;
  std::vector<double> batchMean, batchVar;

  CNN::BatchNorm<double>::propagate(batch, shape, params, config, true, &xNorm, &batchMean, &batchVar);

  // Mean across all 4 values: (1+3+5+7)/4 = 4.0
  CHECK_NEAR(batchMean[0], 4.0, 1e-9, "batchnorm train mean");

  // Var: ((1-4)^2 + (3-4)^2 + (5-4)^2 + (7-4)^2) / 4 = (9+1+1+9)/4 = 5.0
  CHECK_NEAR(batchVar[0], 5.0, 1e-9, "batchnorm train var");

  // With gamma=1, beta=0: output = xNormalized
  double invStd = 1.0 / std::sqrt(5.0);
  CHECK_NEAR(s0.data[0], (1.0 - 4.0) * invStd, 1e-9, "batchnorm train s0[0]");
  CHECK_NEAR(s0.data[1], (3.0 - 4.0) * invStd, 1e-9, "batchnorm train s0[1]");
  CHECK_NEAR(s1.data[0], (5.0 - 4.0) * invStd, 1e-9, "batchnorm train s1[0]");
  CHECK_NEAR(s1.data[1], (7.0 - 4.0) * invStd, 1e-9, "batchnorm train s1[1]");

  // Running stats: runningMean = 0.9*0 + 0.1*4 = 0.4
  CHECK_NEAR(params.runningMean[0], 0.4, 1e-7, "batchnorm train runningMean");
  // runningVar = 0.9*1 + 0.1*5 = 1.4
  CHECK_NEAR(params.runningVar[0], 1.4, 1e-7, "batchnorm train runningVar");
}

//===================================================================================================================//

static void testBatchNormBackpropagate()
{
  std::cout << "--- testBatchNormBackpropagate ---" << std::endl;

  // 1 channel, 1x2 spatial, 2 samples
  CNN::Shape3D shape{1, 1, 2};

  CNN::Tensor3D<double> s0(shape);
  s0.data = {1.0, 3.0};

  CNN::Tensor3D<double> s1(shape);
  s1.data = {5.0, 7.0};

  std::vector<CNN::Tensor3D<double>*> batch = {&s0, &s1};

  CNN::NormParameters<double> params;
  params.numChannels = 1;
  params.gamma = {2.0};
  params.beta = {0.5};
  params.runningMean = {0.0};
  params.runningVar = {1.0};

  CNN::NormLayerConfig config;
  config.epsilon = 0.0;
  config.momentum = 0.1;

  std::vector<CNN::Tensor3D<double>> xNorm;
  std::vector<double> batchMean, batchVar;

  CNN::BatchNorm<double>::propagate(batch, shape, params, config, true, &xNorm, &batchMean, &batchVar);

  // Upstream gradient: all ones
  CNN::Tensor3D<double> d0(shape, 1.0);
  CNN::Tensor3D<double> d1(shape, 1.0);
  std::vector<CNN::Tensor3D<double>*> dBatch = {&d0, &d1};

  std::vector<double> dGamma, dBeta;
  CNN::BatchNorm<double>::backpropagate(dBatch, shape, params, config, batchMean, batchVar, xNorm, dGamma, dBeta);

  // dBeta = sum of all dOutputs = 4.0
  CHECK_NEAR(dBeta[0], 4.0, 1e-9, "batchnorm backprop dBeta");

  // dGamma = sum(dOutput * xNorm) = sum(xNorm) = 0 (by construction, xNorm sums to 0)
  CHECK_NEAR(dGamma[0], 0.0, 1e-9, "batchnorm backprop dGamma");

  // With uniform dOutput=1 and dGamma=0: dInput should be 0
  for (ulong i = 0; i < 2; i++) {
    CHECK_NEAR(d0.data[i], 0.0, 1e-9, "batchnorm backprop dInput s0 uniform=0");
    CHECK_NEAR(d1.data[i], 0.0, 1e-9, "batchnorm backprop dInput s1 uniform=0");
  }
}

//===================================================================================================================//

static void testBatchNormValidateShapes()
{
  std::cout << "--- testBatchNormValidateShapes ---" << std::endl;

  CNN::LayersConfig lc;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig bn;
  bn.type = CNN::LayerType::BATCHNORM;
  bn.config = CNN::NormLayerConfig{};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flatten;
  flatten.type = CNN::LayerType::FLATTEN;
  flatten.config = CNN::FlattenLayerConfig{};

  lc.cnnLayers = {conv1, bn, relu1, flatten};

  CNN::Shape3D outShape = lc.validateShapes({1, 8, 8});
  CHECK(outShape.c == 4 && outShape.h == 6 && outShape.w == 6, "validateShapes with batchnorm");
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

void runLayerTests2()
{
  testInstanceNormOutputShape();
  testInstanceNormValidateShapes();
  testBatchNormInference();
  testBatchNormTraining();
  testBatchNormBackpropagate();
  testBatchNormValidateShapes();
  testSlidingStrategy();
  testDeviceNameToType();
  testModeNameToType();
  testPoolTypeNameToType();
  testCostFunctionNameToType();
  testValidateShapes();
}
