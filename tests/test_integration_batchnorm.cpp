#include "test_helpers.hpp"

//===================================================================================================================//

// Helper: build a Conv→BN→ReLU→Flatten→Dense config with preset parameters
static CNN::CoreConfig<double> makeBNTestConfig(ulong denseNeurons, ANN::ActvFuncType actvFunc)
{
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 3, 3};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numThreads = 1;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig bnLayer;
  bnLayer.type = CNN::LayerType::INSTANCENORM;
  bnLayer.config = CNN::NormLayerConfig{1e-5f, 0.1f};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, bnLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{denseNeurons, actvFunc}};

  // Preset conv parameters
  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 2;
  initConv.filterW = 2;
  initConv.filters = {0.1, -0.2, 0.3, -0.1};
  initConv.biases = {0.0};
  config.parameters.convParams = {initConv};

  // Preset BN parameters: 1 channel, gamma=1, beta=0, runningMean=0, runningVar=1
  CNN::NormParameters<double> initBN;
  initBN.numChannels = 1;
  initBN.gamma = {1.0};
  initBN.beta = {0.0};
  initBN.runningMean = {0.0};
  initBN.runningVar = {1.0};
  config.parameters.normParams = {initBN};

  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.learningRate = 1.0f;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;

  return config;
}

//===================================================================================================================//

static void testExactBNForwardBackwardCrossEntropy()
{
  std::cout << "--- testExactBNForwardBackwardCrossEntropy ---" << std::endl;

  auto config = makeBNTestConfig(2, ANN::ActvFuncType::SOFTMAX);
  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;

  // Dense: 4 inputs → 2 softmax outputs
  ANN::Parameters<double> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1, -0.2, 0.3, -0.1}, {0.2, 0.1, -0.3, 0.2}};
  denseParams.biases[1] = {0.0, 0.0};
  config.parameters.denseParams = denseParams;

  CNN::Samples<double> samples(1);
  samples[0].input = CNN::Tensor3D<double>({1, 3, 3});
  samples[0].input.data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  samples[0].output = {1.0, 0.0};

  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<double>& p = core->getParameters();

  // Conv filter
  CHECK_NEAR(p.convParams[0].filters[0], 0.10000000000000089, 1e-14, "BN-CE conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.19999999999999912, 1e-14, "BN-CE conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.29999999999999999, 1e-14, "BN-CE conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.099999999999998229, 1e-14, "BN-CE conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], 1.7763568394002505e-15, 1e-14, "BN-CE conv bias");

  // InstanceNorm parameters
  CHECK_NEAR(p.normParams[0].gamma[0], 1.0, 1e-14, "BN-CE gamma");
  CHECK_NEAR(p.normParams[0].beta[0], 0.14999999999999997, 1e-14, "BN-CE beta");
  CHECK_NEAR(p.normParams[0].runningMean[0], 0.0060000000894069655, 1e-14, "BN-CE runningMean");
  CHECK_NEAR(p.normParams[0].runningVar[0], 0.90002499851025641, 1e-14, "BN-CE runningVar");

  // Dense weights
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.10000000000000001, 1e-14, "BN-CE dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.20000000000000001, 1e-14, "BN-CE dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.61008683662366447, 1e-14, "BN-CE dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], 0.52017367324732899, 1e-14, "BN-CE dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 0.5, 1e-14, "BN-CE db[0]");
  CHECK_NEAR(p.denseParams.weights[1][1][0], 0.20000000000000001, 1e-14, "BN-CE dw[1][0]");
  CHECK_NEAR(p.denseParams.weights[1][1][1], 0.10000000000000001, 1e-14, "BN-CE dw[1][1]");
  CHECK_NEAR(p.denseParams.weights[1][1][2], -0.61008683662366447, 1e-14, "BN-CE dw[1][2]");
  CHECK_NEAR(p.denseParams.weights[1][1][3], -0.42017367324732896, 1e-14, "BN-CE dw[1][3]");
  CHECK_NEAR(p.denseParams.biases[1][1], -0.5, 1e-14, "BN-CE db[1]");
}

//===================================================================================================================//

static void testExactBNForwardBackwardSquaredDifference()
{
  std::cout << "--- testExactBNForwardBackwardSquaredDifference ---" << std::endl;

  auto config = makeBNTestConfig(1, ANN::ActvFuncType::SIGMOID);
  config.costFunctionConfig.type = CNN::CostFunctionType::SQUARED_DIFFERENCE;

  ANN::Parameters<double> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1, -0.2, 0.3, -0.1}};
  denseParams.biases[1] = {0.0};
  config.parameters.denseParams = denseParams;

  CNN::Samples<double> samples(1);
  samples[0].input = CNN::Tensor3D<double>({1, 3, 3});
  samples[0].input.data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  samples[0].output = {1.0};

  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<double>& p = core->getParameters();

  // Conv filter
  CHECK_NEAR(p.convParams[0].filters[0], 0.10577280381841839, 1e-14, "BN-SD conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.19422719618158163, 1e-14, "BN-SD conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.30577280381841793, 1e-14, "BN-SD conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.094227196181581624, 1e-14, "BN-SD conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], 0.0, 1e-14, "BN-SD conv bias");

  // InstanceNorm parameters
  CHECK_NEAR(p.normParams[0].gamma[0], 1.015009290292471, 1e-14, "BN-SD gamma");
  CHECK_NEAR(p.normParams[0].beta[0], 0.048403506759260036, 1e-14, "BN-SD beta");
  CHECK_NEAR(p.normParams[0].runningMean[0], 0.0060000000894069655, 1e-14, "BN-SD runningMean");
  CHECK_NEAR(p.normParams[0].runningVar[0], 0.90002499851025641, 1e-14, "BN-SD runningVar");

  // Dense weights
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.10000000000000001, 1e-14, "BN-SD dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.20000000000000001, 1e-14, "BN-SD dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.45009290292471105, 1e-14, "BN-SD dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], 0.20018580584942217, 1e-14, "BN-SD dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 0.2420175337963002, 1e-14, "BN-SD db[0]");
}

//===================================================================================================================//

static void testExactBNForwardBackwardWeightedCrossEntropy()
{
  std::cout << "--- testExactBNForwardBackwardWeightedCrossEntropy ---" << std::endl;

  auto config = makeBNTestConfig(2, ANN::ActvFuncType::SOFTMAX);
  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {3.0, 0.5};

  ANN::Parameters<double> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1, -0.2, 0.3, -0.1}, {0.2, 0.1, -0.3, 0.2}};
  denseParams.biases[1] = {0.0, 0.0};
  config.parameters.denseParams = denseParams;

  CNN::Samples<double> samples(1);
  samples[0].input = CNN::Tensor3D<double>({1, 3, 3});
  samples[0].input.data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  samples[0].output = {1.0, 0.0};

  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<double>& p = core->getParameters();

  // Conv filter
  CHECK_NEAR(p.convParams[0].filters[0], 0.10000000000000001, 1e-14, "BN-WCE conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.20000000000000356, 1e-14, "BN-WCE conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.29999999999998933, 1e-14, "BN-WCE conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.10000000000000001, 1e-14, "BN-WCE conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], 0.0, 1e-14, "BN-WCE conv bias");

  // InstanceNorm parameters
  CHECK_NEAR(p.normParams[0].gamma[0], 0.99999999999999989, 1e-14, "BN-WCE gamma");
  CHECK_NEAR(p.normParams[0].beta[0], 0.44999999999999984, 1e-14, "BN-WCE beta");
  CHECK_NEAR(p.normParams[0].runningMean[0], 0.0060000000894069655, 1e-14, "BN-WCE runningMean");
  CHECK_NEAR(p.normParams[0].runningVar[0], 0.90002499851025641, 1e-14, "BN-WCE runningVar");

  // Dense weights
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.10000000000000001, 1e-14, "BN-WCE dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.20000000000000001, 1e-14, "BN-WCE dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 1.2302605098709936, 1e-14, "BN-WCE dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], 1.7605210197419869, 1e-14, "BN-WCE dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 1.5, 1e-14, "BN-WCE db[0]");
  CHECK_NEAR(p.denseParams.weights[1][1][0], 0.20000000000000001, 1e-14, "BN-WCE dw[1][0]");
  CHECK_NEAR(p.denseParams.weights[1][1][1], 0.10000000000000001, 1e-14, "BN-WCE dw[1][1]");
  CHECK_NEAR(p.denseParams.weights[1][1][2], -1.2302605098709936, 1e-14, "BN-WCE dw[1][2]");
  CHECK_NEAR(p.denseParams.weights[1][1][3], -1.6605210197419871, 1e-14, "BN-WCE dw[1][3]");
  CHECK_NEAR(p.denseParams.biases[1][1], -1.5, 1e-14, "BN-WCE db[1]");
}

void runIntegrationBatchNormTests()
{
  testExactBNForwardBackwardCrossEntropy();
  testExactBNForwardBackwardSquaredDifference();
  testExactBNForwardBackwardWeightedCrossEntropy();
}
