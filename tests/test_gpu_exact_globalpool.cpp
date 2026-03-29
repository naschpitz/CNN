#include "test_helpers.hpp"

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

static void testGPUGlobalDualPoolCPUGPUParity()
{
  std::cout << "--- testGPUGlobalDualPoolCPUGPUParity ---" << std::endl;

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

    CNN::CNNLayerConfig gdpLayer;
    gdpLayer.type = CNN::LayerType::GLOBALDUALPOOL;
    gdpLayer.config = CNN::GlobalDualPoolLayerConfig{};

    CNN::CNNLayerConfig flattenLayer;
    flattenLayer.type = CNN::LayerType::FLATTEN;
    flattenLayer.config = CNN::FlattenLayerConfig{};

    // 2 conv filters -> GDP outputs 4 features -> dense 1
    config.layersConfig.cnnLayers = {conv1, relu1, gdpLayer, flattenLayer};
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

    // Preset dense weights: input=4 (2 avg + 2 max), output=1
    config.parameters.denseParams.weights.resize(2);
    config.parameters.denseParams.weights[1] = {{0.1f, -0.2f, 0.15f, -0.1f}};
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

  CHECK_NEAR(cpuPred0[0], gpuPred0[0], 1e-3, "gdp CPU-GPU parity pred0");
  CHECK_NEAR(cpuPred1[0], gpuPred1[0], 1e-3, "gdp CPU-GPU parity pred1");
}

//===================================================================================================================//


//===================================================================================================================//

void runGPUExactGlobalPoolTests()
{
  testGPUGlobalAvgPoolCPUGPUParity();
  testGPUGlobalDualPoolCPUGPUParity();
}
