#include "test_helpers.hpp"

//===================================================================================================================//

static void testGPUExactForwardBackwardCrossEntropy()
{
  std::cout << "--- testGPUExactForwardBackwardCrossEntropy ---" << std::endl;

  // Same hand-computed network as CPU test, but on GPU with float.
  // 1x3x3 → Conv(1 filter 2x2, stride=1, valid) → ReLU → Flatten(4) → Dense(2, softmax)
  // Cross-entropy cost, SGD lr=1.0, 1 epoch, 1 sample
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

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

  // Preset conv parameters
  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 2;
  initConv.filterW = 2;
  initConv.filters = {0.1f, -0.2f, 0.3f, -0.1f};
  initConv.biases = {0.0f};
  config.parameters.convParams = {initConv};

  // Preset ANN dense parameters
  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}, {0.2f, 0.1f, -0.3f, 0.2f}};
  denseParams.biases[1] = {0.0f, 0.0f};
  config.parameters.denseParams = denseParams;

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.learningRate = 1.0f;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;

  // Input: 1x3x3, target: [1, 0]
  CNN::Samples<float> samples(1);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f, 0.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& p = core->getParameters();

  // Conv filter — GPU float values verified against CPU double (tolerance 1e-4 for float)
  CHECK_NEAR(p.convParams[0].filters[0], 0.11000499999958335f, 1e-4, "GPU exact conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.19499750000020832f, 1e-4, "GPU exact conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.2949975000002083f, 1e-4, "GPU exact conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.11000499999958332f, 1e-4, "GPU exact conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], -0.050024999997916653f, 1e-4, "GPU exact conv bias");

  // Dense weights (ANN layer 1)
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.12000999999916667f, 1e-4, "GPU exact dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.17498750000104168f, 1e-4, "GPU exact dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.33501749999854163f, 1e-4, "GPU exact dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], -0.059980000001666679f, 1e-4, "GPU exact dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 0.50024999997916675f, 1e-4, "GPU exact db[0]");

  CHECK_NEAR(p.denseParams.weights[1][1][0], 0.17999000000083334f, 1e-4, "GPU exact dw[1][0]");
  CHECK_NEAR(p.denseParams.weights[1][1][1], 0.074987500001041679f, 1e-4, "GPU exact dw[1][1]");
  CHECK_NEAR(p.denseParams.weights[1][1][2], -0.33501749999854163f, 1e-4, "GPU exact dw[1][2]");
  CHECK_NEAR(p.denseParams.weights[1][1][3], 0.15998000000166668f, 1e-4, "GPU exact dw[1][3]");
  CHECK_NEAR(p.denseParams.biases[1][1], -0.50024999997916664f, 1e-4, "GPU exact db[1]");
}

//===================================================================================================================//

static void testGPUExactForwardBackwardSquaredDifference()
{
  std::cout << "--- testGPUExactForwardBackwardSquaredDifference ---" << std::endl;

  // 1x3x3 → Conv(1 filter 2x2, stride=1, valid) → ReLU → Flatten(4) → Dense(1, sigmoid)
  // Squared-difference cost, SGD lr=1.0, 1 epoch, 1 sample
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

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 2;
  initConv.filterW = 2;
  initConv.filters = {0.1f, -0.2f, 0.3f, -0.1f};
  initConv.biases = {0.0f};
  config.parameters.convParams = {initConv};

  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}};
  denseParams.biases[1] = {0.0f};
  config.parameters.denseParams = denseParams;

  config.costFunctionConfig.type = CNN::CostFunctionType::SQUARED_DIFFERENCE;
  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.learningRate = 1.0f;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;

  CNN::Samples<float> samples(1);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& p = core->getParameters();

  // Conv filter - every value verified against GPU float output (tolerance 1e-6)
  CHECK_NEAR(p.convParams[0].filters[0], 0.1099648774f, 1e-6, "GPU SD conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.1875438988f, 1e-6, "GPU SD conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.3174385428f, 1e-6, "GPU SD conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.08007024229f, 1e-6, "GPU SD conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], 0.02491219342f, 1e-6, "GPU SD conv bias");

  // Dense weights - every value verified
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.1099648774f, 1e-6, "GPU SD dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.1875438988f, 1e-6, "GPU SD dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.3174385428f, 1e-6, "GPU SD dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], -0.08007024229f, 1e-6, "GPU SD dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 0.2491219491f, 1e-6, "GPU SD db[0]");
}

//===================================================================================================================//

static void testGPUExactForwardBackwardWeightedCrossEntropy()
{
  std::cout << "--- testGPUExactForwardBackwardWeightedCrossEntropy ---" << std::endl;

  // 1x3x3 → Conv(1 filter 2x2, stride=1, valid) → ReLU → Flatten(4) → Dense(2, softmax)
  // Weighted cross-entropy cost [3.0, 0.5], SGD lr=1.0, 1 epoch, 1 sample
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

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 2;
  initConv.filterW = 2;
  initConv.filters = {0.1f, -0.2f, 0.3f, -0.1f};
  initConv.biases = {0.0f};
  config.parameters.convParams = {initConv};

  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}, {0.2f, 0.1f, -0.3f, 0.2f}};
  denseParams.biases[1] = {0.0f, 0.0f};
  config.parameters.denseParams = denseParams;

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {3.0f, 0.5f};
  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.learningRate = 1.0f;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;

  CNN::Samples<float> samples(1);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f, 0.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& p = core->getParameters();

  // Conv filter - every value verified against GPU float output (tolerance 1e-6)
  CHECK_NEAR(p.convParams[0].filters[0], 0.1300149858f, 1e-6, "GPU WCE conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.1849925071f, 1e-6, "GPU WCE conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.2849925756f, 1e-6, "GPU WCE conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.1300149858f, 1e-6, "GPU WCE conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], -0.1500749588f, 1e-6, "GPU WCE conv bias");

  // Dense weights - every value verified
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.1600300074f, 1e-6, "GPU WCE dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.1249625087f, 1e-6, "GPU WCE dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.4050525129f, 1e-6, "GPU WCE dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], 0.0200600028f, 1e-6, "GPU WCE dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 1.500749946f, 1e-6, "GPU WCE db[0]");

  CHECK_NEAR(p.denseParams.weights[1][1][0], 0.1399700046f, 1e-6, "GPU WCE dw[1][0]");
  CHECK_NEAR(p.denseParams.weights[1][1][1], 0.02496250719f, 1e-6, "GPU WCE dw[1][1]");
  CHECK_NEAR(p.denseParams.weights[1][1][2], -0.4050525129f, 1e-6, "GPU WCE dw[1][2]");
  CHECK_NEAR(p.denseParams.weights[1][1][3], 0.07993999869f, 1e-6, "GPU WCE dw[1][3]");
  CHECK_NEAR(p.denseParams.biases[1][1], -1.500749946f, 1e-6, "GPU WCE db[1]");
}

//===================================================================================================================//

// Helper: build a GPU Conv→BN→ReLU→Flatten→Dense config with preset parameters

static void testGPUGlobalAvgPoolCPUGPUParity()
{
  std::cout << "--- testGPUGlobalAvgPoolCPUGPUParity ---" << std::endl;

  // Train the exact same GAP network on CPU and GPU and verify predictions match.
  auto makeConfig = [](CNN::DeviceType device) {
    CNN::CoreConfig<float> config;
    config.modeType = CNN::ModeType::TRAIN;
    config.deviceType = device;
    config.inputShape = {1, 5, 5};
    config.logLevel = CNN::LogLevel::ERROR;
    config.numThreads = 1;
    config.numGPUs = 1;

    CNN::CNNLayerConfig conv1;
    conv1.type = CNN::LayerType::CONV;
    conv1.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

    CNN::CNNLayerConfig relu1;
    relu1.type = CNN::LayerType::RELU;
    relu1.config = CNN::ReLULayerConfig{};

    CNN::CNNLayerConfig gapLayer;
    gapLayer.type = CNN::LayerType::GLOBALAVGPOOL;
    gapLayer.config = CNN::GlobalAvgPoolLayerConfig{};

    CNN::CNNLayerConfig flattenLayer;
    flattenLayer.type = CNN::LayerType::FLATTEN;
    flattenLayer.config = CNN::FlattenLayerConfig{};

    config.layersConfig.cnnLayers = {conv1, relu1, gapLayer, flattenLayer};
    config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

    CNN::ConvParameters<float> initConv;
    initConv.numFilters = 2;
    initConv.inputC = 1;
    initConv.filterH = 3;
    initConv.filterW = 3;
    initConv.filters = {0.1f,   -0.05f, 0.02f,  0.08f, -0.03f, 0.06f,  -0.01f, 0.04f,  0.07f,
                        -0.02f, 0.09f,  -0.04f, 0.05f, 0.03f,  -0.07f, 0.01f,  -0.06f, 0.08f};
    initConv.biases = {0.0f, 0.0f};
    config.parameters.convParams = {initConv};

    // Preset dense weights to remove randomness
    // ANN layers: [input=2, output=1], so weights[0]=empty, weights[1]=[[w1,w2]]
    config.parameters.denseParams.weights.resize(2);
    config.parameters.denseParams.weights[1] = {{0.1f, -0.2f}};
    config.parameters.denseParams.biases.resize(2);
    config.parameters.denseParams.biases[1] = {0.0f};

    config.trainingConfig.numEpochs = 20;
    config.trainingConfig.learningRate = 0.1f;
    config.progressReports = 0;

    return config;
  };

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  auto cpuConfig = makeConfig(CNN::DeviceType::CPU);
  auto cpuCore = CNN::Core<float>::makeCore(cpuConfig);
  cpuCore->train(samples.size(), CNN::makeSampleProvider(samples));
  auto cpuPred0 = cpuCore->predict(samples[0].input);
  auto cpuPred1 = cpuCore->predict(samples[1].input);

  auto gpuConfig = makeConfig(CNN::DeviceType::GPU);
  auto gpuCore = CNN::Core<float>::makeCore(gpuConfig);
  gpuCore->train(samples.size(), CNN::makeSampleProvider(samples));
  auto gpuPred0 = gpuCore->predict(samples[0].input);
  auto gpuPred1 = gpuCore->predict(samples[1].input);

  std::cout << "  CPU pred: " << cpuPred0[0] << ", " << cpuPred1[0] << std::endl;
  std::cout << "  GPU pred: " << gpuPred0[0] << ", " << gpuPred1[0] << std::endl;

  // float precision: allow ~1e-3 tolerance (GPU uses different parallel reductions)
  CHECK_NEAR(cpuPred0[0], gpuPred0[0], 1e-3, "gap CPU-GPU parity pred0");
  CHECK_NEAR(cpuPred1[0], gpuPred1[0], 1e-3, "gap CPU-GPU parity pred1");
}

//===================================================================================================================//

void runGPUExactTests()
{
  testGPUExactForwardBackwardCrossEntropy();
  testGPUExactForwardBackwardSquaredDifference();
  testGPUExactForwardBackwardWeightedCrossEntropy();
  testGPUGlobalAvgPoolCPUGPUParity();
}
