#include "test_helpers.hpp"

//===================================================================================================================//

static void testExactForwardBackwardCrossEntropy()
{
  std::cout << "--- testExactForwardBackwardCrossEntropy ---" << std::endl;

  // 1x3x3 → Conv(1 filter 2x2, stride=1, valid) → ReLU → Flatten(4) → Dense(2, softmax)
  // Cross-entropy cost, SGD lr=1.0, 1 epoch, 1 sample, no shuffle, 1 thread
  // Every single parameter is preset and verified against hand-computed values.

  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 3, 3};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numThreads = 1;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

  // Preset conv parameters: 1 filter, 1 channel, 2x2
  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 2;
  initConv.filterW = 2;
  initConv.filters = {0.1, -0.2, 0.3, -0.1};
  initConv.biases = {0.0};
  config.parameters.convParams = {initConv};

  // Preset ANN dense parameters: layer 0 (input, 4 neurons), layer 1 (softmax, 2 neurons)
  ANN::Parameters<double> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {}; // input layer (unused)
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1, -0.2, 0.3, -0.1}, {0.2, 0.1, -0.3, 0.2}};
  denseParams.biases[1] = {0.0, 0.0};
  config.parameters.denseParams = denseParams;

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;

  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.learningRate = 1.0f;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;

  // Input: 1x3x3, target: [1, 0]
  CNN::Samples<double> samples(1);
  samples[0].input = CNN::Tensor3D<double>({1, 3, 3});
  samples[0].input.data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  samples[0].output = {1.0, 0.0};

  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<double>& p = core->getParameters();

  // Conv filter (updated by SGD) - values from C++ output
  CHECK_NEAR(p.convParams[0].filters[0], 0.11000499999958335, 1e-14, "exact conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.19499750000020832, 1e-14, "exact conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.2949975000002083, 1e-14, "exact conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.11000499999958332, 1e-14, "exact conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], -0.050024999997916653, 1e-14, "exact conv bias");

  // Dense weights (ANN layer 1) - values from C++ output
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.12000999999916667, 1e-14, "exact dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.17498750000104168, 1e-14, "exact dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.33501749999854163, 1e-14, "exact dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], -0.059980000001666679, 1e-14, "exact dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 0.50024999997916675, 1e-14, "exact db[0]");

  CHECK_NEAR(p.denseParams.weights[1][1][0], 0.17999000000083334, 1e-14, "exact dw[1][0]");
  CHECK_NEAR(p.denseParams.weights[1][1][1], 0.074987500001041679, 1e-14, "exact dw[1][1]");
  CHECK_NEAR(p.denseParams.weights[1][1][2], -0.33501749999854163, 1e-14, "exact dw[1][2]");
  CHECK_NEAR(p.denseParams.weights[1][1][3], 0.15998000000166668, 1e-14, "exact dw[1][3]");
  CHECK_NEAR(p.denseParams.biases[1][1], -0.50024999997916664, 1e-14, "exact db[1]");
}

//===================================================================================================================//

static void testExactForwardBackwardSquaredDifference()
{
  std::cout << "--- testExactForwardBackwardSquaredDifference ---" << std::endl;

  // 1x3x3 → Conv(1 filter 2x2, stride=1, valid) → ReLU → Flatten(4) → Dense(1, sigmoid)
  // Squared-difference cost, SGD lr=1.0, 1 epoch, 1 sample, no shuffle, 1 thread
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 3, 3};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numThreads = 1;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  // Preset conv parameters: 1 filter, 1 channel, 2x2
  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 2;
  initConv.filterW = 2;
  initConv.filters = {0.1, -0.2, 0.3, -0.1};
  initConv.biases = {0.0};
  config.parameters.convParams = {initConv};

  // Preset ANN dense parameters: layer 0 (input, 4 neurons), layer 1 (sigmoid, 1 neuron)
  ANN::Parameters<double> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1, -0.2, 0.3, -0.1}};
  denseParams.biases[1] = {0.0};
  config.parameters.denseParams = denseParams;

  config.costFunctionConfig.type = CNN::CostFunctionType::SQUARED_DIFFERENCE;

  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.learningRate = 1.0f;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;

  // Input: 1x3x3, target: [1.0]
  CNN::Samples<double> samples(1);
  samples[0].input = CNN::Tensor3D<double>({1, 3, 3});
  samples[0].input.data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  samples[0].output = {1.0};

  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<double>& p = core->getParameters();

  // Conv filter - every value verified against C++ output
  CHECK_NEAR(p.convParams[0].filters[0], 0.10996487779676728, 1e-14, "SD conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.18754390275404093, 1e-14, "SD conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.3174385361443427, 1e-14, "SD conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.080070244406465471, 1e-14, "SD conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], 0.024912194491918171, 1e-14, "SD conv bias");

  // Dense weights (ANN layer 1) - every value verified
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.10996487779676728, 1e-14, "SD dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.18754390275404093, 1e-14, "SD dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.3174385361443427, 1e-14, "SD dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], -0.080070244406465471, 1e-14, "SD dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 0.24912194491918171, 1e-14, "SD db[0]");
}

//===================================================================================================================//

static void testExactForwardBackwardWeightedCrossEntropy()
{
  std::cout << "--- testExactForwardBackwardWeightedCrossEntropy ---" << std::endl;

  // 1x3x3 → Conv(1 filter 2x2, stride=1, valid) → ReLU → Flatten(4) → Dense(2, softmax)
  // Weighted cross-entropy cost [3.0, 0.5], SGD lr=1.0, 1 epoch, 1 sample
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 3, 3};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numThreads = 1;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 2;
  initConv.filterW = 2;
  initConv.filters = {0.1, -0.2, 0.3, -0.1};
  initConv.biases = {0.0};
  config.parameters.convParams = {initConv};

  ANN::Parameters<double> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1, -0.2, 0.3, -0.1}, {0.2, 0.1, -0.3, 0.2}};
  denseParams.biases[1] = {0.0, 0.0};
  config.parameters.denseParams = denseParams;

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {3.0, 0.5};

  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.learningRate = 1.0f;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;

  CNN::Samples<double> samples(1);
  samples[0].input = CNN::Tensor3D<double>({1, 3, 3});
  samples[0].input.data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  samples[0].output = {1.0, 0.0};

  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<double>& p = core->getParameters();

  // Conv filter - every value verified against C++ output
  CHECK_NEAR(p.convParams[0].filters[0], 0.13001499999874996, 1e-14, "WCE conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.184992500000625, 1e-14, "WCE conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.28499250000062482, 1e-14, "WCE conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.13001499999875007, 1e-14, "WCE conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], -0.15007499999375007, 1e-14, "WCE conv bias");

  // Dense weights (ANN layer 1) - every value verified
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.16002999999750001, 1e-14, "WCE dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.12496250000312502, 1e-14, "WCE dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.40505249999562493, 1e-14, "WCE dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], 0.020059999994999939, 1e-14, "WCE dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 1.5007499999374998, 1e-14, "WCE db[0]");

  CHECK_NEAR(p.denseParams.weights[1][1][0], 0.13997000000250004, 1e-14, "WCE dw[1][0]");
  CHECK_NEAR(p.denseParams.weights[1][1][1], 0.024962500003125013, 1e-14, "WCE dw[1][1]");
  CHECK_NEAR(p.denseParams.weights[1][1][2], -0.40505249999562493, 1e-14, "WCE dw[1][2]");
  CHECK_NEAR(p.denseParams.weights[1][1][3], 0.079940000005000067, 1e-14, "WCE dw[1][3]");
  CHECK_NEAR(p.denseParams.biases[1][1], -1.5007499999374998, 1e-14, "WCE db[1]");
}

void runIntegrationExactTests()
{
  testExactForwardBackwardCrossEntropy();
  testExactForwardBackwardSquaredDifference();
  testExactForwardBackwardWeightedCrossEntropy();
}
