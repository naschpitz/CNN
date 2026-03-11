#include "test_helpers.hpp"

static CNN::CoreConfig<float> makeGPUBNTestConfig(ulong denseNeurons, ANN::ActvFuncType actvFunc)
{
  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 3, 3};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numThreads = 1;
  config.numGPUs = 1;

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

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 2;
  initConv.filterW = 2;
  initConv.filters = {0.1f, -0.2f, 0.3f, -0.1f};
  initConv.biases = {0.0f};
  config.parameters.convParams = {initConv};

  CNN::NormParameters<float> initBN;
  initBN.numChannels = 1;
  initBN.gamma = {1.0f};
  initBN.beta = {0.0f};
  initBN.runningMean = {0.0f};
  initBN.runningVar = {1.0f};
  config.parameters.normParams = {initBN};

  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.learningRate = 1.0f;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;

  return config;
}

static void testGPUExactBNForwardBackwardCrossEntropy()
{
  std::cout << "--- testGPUExactBNForwardBackwardCrossEntropy ---" << std::endl;

  auto config = makeGPUBNTestConfig(2, ANN::ActvFuncType::SOFTMAX);
  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;

  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}, {0.2f, 0.1f, -0.3f, 0.2f}};
  denseParams.biases[1] = {0.0f, 0.0f};
  config.parameters.denseParams = denseParams;

  CNN::Samples<float> samples(1);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f, 0.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& p = core->getParameters();

  // Conv filter
  CHECK_NEAR(p.convParams[0].filters[0], 0.1000009552f, 1e-6, "GPU BN-CE filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.1999985725f, 1e-6, "GPU BN-CE filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.3000019193f, 1e-6, "GPU BN-CE filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.09999809414f, 1e-6, "GPU BN-CE filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], 9.536743164e-07f, 1e-5, "GPU BN-CE conv bias");

  // InstanceNorm parameters
  CHECK_NEAR(p.normParams[0].gamma[0], 0.9999999404f, 1e-6, "GPU BN-CE gamma");
  CHECK_NEAR(p.normParams[0].beta[0], 0.150000006f, 1e-6, "GPU BN-CE beta");
  CHECK_NEAR(p.normParams[0].runningMean[0], 0.006000000052f, 1e-6, "GPU BN-CE runningMean");
  CHECK_NEAR(p.normParams[0].runningVar[0], 0.9000249505f, 1e-6, "GPU BN-CE runningVar");

  // Dense weights
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.1000000015f, 1e-6, "GPU BN-CE dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.200000003f, 1e-6, "GPU BN-CE dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.6100869179f, 1e-6, "GPU BN-CE dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], 0.5201738477f, 1e-6, "GPU BN-CE dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 0.5f, 1e-6, "GPU BN-CE db[0]");
  CHECK_NEAR(p.denseParams.weights[1][1][0], 0.200000003f, 1e-6, "GPU BN-CE dw[1][0]");
  CHECK_NEAR(p.denseParams.weights[1][1][1], 0.1000000015f, 1e-6, "GPU BN-CE dw[1][1]");
  CHECK_NEAR(p.denseParams.weights[1][1][2], -0.6100869179f, 1e-6, "GPU BN-CE dw[1][2]");
  CHECK_NEAR(p.denseParams.weights[1][1][3], -0.4201738834f, 1e-6, "GPU BN-CE dw[1][3]");
  CHECK_NEAR(p.denseParams.biases[1][1], -0.5f, 1e-6, "GPU BN-CE db[1]");
}

//===================================================================================================================//

static void testGPUExactBNForwardBackwardSquaredDifference()
{
  std::cout << "--- testGPUExactBNForwardBackwardSquaredDifference ---" << std::endl;

  auto config = makeGPUBNTestConfig(1, ANN::ActvFuncType::SIGMOID);
  config.costFunctionConfig.type = CNN::CostFunctionType::SQUARED_DIFFERENCE;

  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}};
  denseParams.biases[1] = {0.0f};
  config.parameters.denseParams = denseParams;

  CNN::Samples<float> samples(1);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& p = core->getParameters();

  // Conv filter
  CHECK_NEAR(p.convParams[0].filters[0], 0.1057728305f, 1e-6, "GPU BN-SD filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.1942272931f, 1e-6, "GPU BN-SD filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.305772841f, 1e-6, "GPU BN-SD filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.09422717243f, 1e-6, "GPU BN-SD filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], 0.0f, 1e-6, "GPU BN-SD conv bias");

  // InstanceNorm parameters
  CHECK_NEAR(p.normParams[0].gamma[0], 1.015009284f, 1e-6, "GPU BN-SD gamma");
  CHECK_NEAR(p.normParams[0].beta[0], 0.04840351269f, 1e-6, "GPU BN-SD beta");
  CHECK_NEAR(p.normParams[0].runningMean[0], 0.006000000052f, 1e-6, "GPU BN-SD runningMean");
  CHECK_NEAR(p.normParams[0].runningVar[0], 0.9000249505f, 1e-6, "GPU BN-SD runningVar");

  // Dense weights
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.1000000015f, 1e-6, "GPU BN-SD dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.200000003f, 1e-6, "GPU BN-SD dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.4500929415f, 1e-6, "GPU BN-SD dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], 0.2001859248f, 1e-6, "GPU BN-SD dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 0.2420175374f, 1e-6, "GPU BN-SD db[0]");
}

//===================================================================================================================//

static void testGPUExactBNForwardBackwardWeightedCrossEntropy()
{
  std::cout << "--- testGPUExactBNForwardBackwardWeightedCrossEntropy ---" << std::endl;

  auto config = makeGPUBNTestConfig(2, ANN::ActvFuncType::SOFTMAX);
  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {3.0f, 0.5f};

  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}, {0.2f, 0.1f, -0.3f, 0.2f}};
  denseParams.biases[1] = {0.0f, 0.0f};
  config.parameters.denseParams = denseParams;

  CNN::Samples<float> samples(1);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f, 0.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& p = core->getParameters();

  // Conv filter
  CHECK_NEAR(p.convParams[0].filters[0], 0.1000019088f, 1e-6, "GPU BN-WCE filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.200000003f, 1e-6, "GPU BN-WCE filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.3000000119f, 1e-6, "GPU BN-WCE filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.1000000015f, 1e-6, "GPU BN-WCE filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], 0.0f, 1e-6, "GPU BN-WCE conv bias");

  // InstanceNorm parameters
  CHECK_NEAR(p.normParams[0].gamma[0], 0.9999998808f, 1e-6, "GPU BN-WCE gamma");
  CHECK_NEAR(p.normParams[0].beta[0], 0.4500000179f, 1e-6, "GPU BN-WCE beta");
  CHECK_NEAR(p.normParams[0].runningMean[0], 0.006000000052f, 1e-6, "GPU BN-WCE runningMean");
  CHECK_NEAR(p.normParams[0].runningVar[0], 0.9000249505f, 1e-6, "GPU BN-WCE runningVar");

  // Dense weights
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.1000000015f, 1e-6, "GPU BN-WCE dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.200000003f, 1e-6, "GPU BN-WCE dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 1.230260611f, 1e-6, "GPU BN-WCE dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], 1.760521531f, 1e-6, "GPU BN-WCE dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 1.5f, 1e-6, "GPU BN-WCE db[0]");
  CHECK_NEAR(p.denseParams.weights[1][1][0], 0.200000003f, 1e-6, "GPU BN-WCE dw[1][0]");
  CHECK_NEAR(p.denseParams.weights[1][1][1], 0.1000000015f, 1e-6, "GPU BN-WCE dw[1][1]");
  CHECK_NEAR(p.denseParams.weights[1][1][2], -1.230260611f, 1e-6, "GPU BN-WCE dw[1][2]");
  CHECK_NEAR(p.denseParams.weights[1][1][3], -1.660521507f, 1e-6, "GPU BN-WCE dw[1][3]");
  CHECK_NEAR(p.denseParams.biases[1][1], -1.5f, 1e-6, "GPU BN-WCE db[1]");
}

void runGPUExactBNTests()
{
  testGPUExactBNForwardBackwardCrossEntropy();
  testGPUExactBNForwardBackwardSquaredDifference();
  testGPUExactBNForwardBackwardWeightedCrossEntropy();
}
