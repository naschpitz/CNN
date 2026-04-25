#include "test_helpers.hpp"

//===================================================================================================================//
//-- True Batch Normalization GPU Tests --//
//===================================================================================================================//

// Helper: build a GPU Conv→BatchNorm→ReLU→Flatten→Dense config with true BATCHNORM
static CNN::CoreConfig<float> makeGPUTrueBNTestConfig(ulong denseNeurons, ANN::ActvFuncType actvFunc,
                                                      CNN::DeviceType deviceType = CNN::DeviceType::GPU)
{
  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = deviceType;
  config.inputShape = {1, 3, 3};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numThreads = 1;
  config.numGPUs = 1;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig bnLayer;
  bnLayer.type = CNN::LayerType::BATCHNORM;
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

//===================================================================================================================//

// Test 1: GPU vs CPU parity with multiple samples (the core test)
// If cross-sample stats are computed correctly, GPU and CPU produce identical results
static void testGPUTrueBNvsCPUParity()
{
  std::cout << "--- testGPUTrueBNvsCPUParity ---" << std::endl;

  // Create identical configs for CPU and GPU
  auto gpuConfig = makeGPUTrueBNTestConfig(2, ANN::ActvFuncType::SOFTMAX, CNN::DeviceType::GPU);
  gpuConfig.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  gpuConfig.trainingConfig.batchSize = 3;

  auto cpuConfig = makeGPUTrueBNTestConfig(2, ANN::ActvFuncType::SOFTMAX, CNN::DeviceType::CPU);
  cpuConfig.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  cpuConfig.trainingConfig.batchSize = 3;

  // Use same dense params for both
  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}, {0.2f, 0.1f, -0.3f, 0.2f}};
  denseParams.biases[1] = {0.0f, 0.0f};
  gpuConfig.parameters.denseParams = denseParams;
  cpuConfig.parameters.denseParams = denseParams;

  // 3 different asymmetric samples to exercise cross-sample statistics.
  // IMPORTANT: avoid symmetric/uniform inputs that produce BN outputs ≈ 0,
  // because tiny GPU/CPU floating-point differences at the RELU boundary
  // cause divergent RELU masks and downstream gradient mismatch.
  CNN::Samples<float> samples(3);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.3f, 0.5f, 0.2f, 0.4f, 0.8f, 0.7f, 0.6f, 0.9f};
  samples[0].output = {1.0f, 0.0f};

  samples[1].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[1].input.data = {0.8f, 0.6f, 0.9f, 0.3f, 0.7f, 0.2f, 0.4f, 0.1f, 0.5f};
  samples[1].output = {0.0f, 1.0f};

  samples[2].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[2].input.data = {0.2f, 0.9f, 0.4f, 0.6f, 0.1f, 0.7f, 0.3f, 0.8f, 0.5f};
  samples[2].output = {1.0f, 0.0f};

  auto gpuCore = CNN::Core<float>::makeCore(gpuConfig);
  auto cpuCore = CNN::Core<float>::makeCore(cpuConfig);

  gpuCore->train(samples.size(), CNN::makeSampleProvider(samples));
  cpuCore->train(samples.size(), CNN::makeSampleProvider(samples));

  const auto& gp = gpuCore->getParameters();
  const auto& cp = cpuCore->getParameters();

  // Conv filters should match
  float tol = 1e-4f;

  for (size_t i = 0; i < 4; i++) {
    CHECK_NEAR(gp.convParams[0].filters[i], cp.convParams[0].filters[i], tol,
               "GPU-CPU BN parity conv filt[" + std::to_string(i) + "]");
  }

  CHECK_NEAR(gp.convParams[0].biases[0], cp.convParams[0].biases[0], tol, "GPU-CPU BN parity conv bias");

  // BN gamma, beta, running stats should match
  CHECK_NEAR(gp.normParams[0].gamma[0], cp.normParams[0].gamma[0], tol, "GPU-CPU BN parity gamma");
  CHECK_NEAR(gp.normParams[0].beta[0], cp.normParams[0].beta[0], tol, "GPU-CPU BN parity beta");
  CHECK_NEAR(gp.normParams[0].runningMean[0], cp.normParams[0].runningMean[0], tol, "GPU-CPU BN parity runningMean");
  CHECK_NEAR(gp.normParams[0].runningVar[0], cp.normParams[0].runningVar[0], tol, "GPU-CPU BN parity runningVar");

  // Dense weights should match
  for (size_t n = 0; n < 2; n++) {
    for (size_t w = 0; w < gp.denseParams.weights[1][n].size(); w++) {
      CHECK_NEAR(gp.denseParams.weights[1][n][w], cp.denseParams.weights[1][n][w], tol,
                 "GPU-CPU BN parity dw[" + std::to_string(n) + "][" + std::to_string(w) + "]");
    }

    CHECK_NEAR(gp.denseParams.biases[1][n], cp.denseParams.biases[1][n], tol,
               "GPU-CPU BN parity db[" + std::to_string(n) + "]");
  }

  // Also check predictions match
  auto gpuPred = gpuCore->predict(samples[0].input).output;
  auto cpuPred = cpuCore->predict(samples[0].input).output;
  CHECK_NEAR(gpuPred[0], cpuPred[0], tol, "GPU-CPU BN parity predict[0]");
  CHECK_NEAR(gpuPred[1], cpuPred[1], tol, "GPU-CPU BN parity predict[1]");

  std::cout << "  GPU predict=[" << gpuPred[0] << "," << gpuPred[1] << "]" << "  CPU predict=[" << cpuPred[0] << ","
            << cpuPred[1] << "]" << std::endl;

  // Debug: print all params
  std::cout << "  GPU conv filt=[" << gp.convParams[0].filters[0] << "," << gp.convParams[0].filters[1] << ","
            << gp.convParams[0].filters[2] << "," << gp.convParams[0].filters[3] << "]"
            << " bias=" << gp.convParams[0].biases[0] << std::endl;
  std::cout << "  CPU conv filt=[" << cp.convParams[0].filters[0] << "," << cp.convParams[0].filters[1] << ","
            << cp.convParams[0].filters[2] << "," << cp.convParams[0].filters[3] << "]"
            << " bias=" << cp.convParams[0].biases[0] << std::endl;
  std::cout << "  GPU gamma=" << gp.normParams[0].gamma[0] << " beta=" << gp.normParams[0].beta[0]
            << " rMean=" << gp.normParams[0].runningMean[0] << " rVar=" << gp.normParams[0].runningVar[0] << std::endl;
  std::cout << "  CPU gamma=" << cp.normParams[0].gamma[0] << " beta=" << cp.normParams[0].beta[0]
            << " rMean=" << cp.normParams[0].runningMean[0] << " rVar=" << cp.normParams[0].runningVar[0] << std::endl;
}

//===================================================================================================================//

// Debug test: Conv → BN → Flatten (no RELU) with N=3 to isolate the issue
static void testGPUTrueBNNoRelu()
{
  std::cout << "--- testGPUTrueBNNoRelu (N=3, no RELU) ---" << std::endl;

  // Build config WITHOUT RELU: Conv → BN → Flatten
  auto makeNoReluConfig = [](CNN::DeviceType device) {
    CNN::CoreConfig<float> config;
    config.modeType = CNN::ModeType::TRAIN;
    config.deviceType = device;
    config.inputShape = {1, 3, 3};
    config.logLevel = CNN::LogLevel::ERROR;
    config.numThreads = 1;
    config.numGPUs = 1;

    CNN::CNNLayerConfig convLayer;
    convLayer.type = CNN::LayerType::CONV;
    convLayer.config = CNN::ConvLayerConfig{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};

    CNN::CNNLayerConfig bnLayer;
    bnLayer.type = CNN::LayerType::BATCHNORM;
    bnLayer.config = CNN::NormLayerConfig{1e-5f, 0.1f};

    CNN::CNNLayerConfig flattenLayer;
    flattenLayer.type = CNN::LayerType::FLATTEN;
    flattenLayer.config = CNN::FlattenLayerConfig{};

    config.layersConfig.cnnLayers = {convLayer, bnLayer, flattenLayer};
    config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

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
    config.trainingConfig.batchSize = 3;
    config.progressReports = 0;
    config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;

    ANN::Parameters<float> denseParams;
    denseParams.weights.resize(2);
    denseParams.biases.resize(2);
    denseParams.weights[0] = {};
    denseParams.biases[0] = {};
    denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}, {0.2f, 0.1f, -0.3f, 0.2f}};
    denseParams.biases[1] = {0.0f, 0.0f};
    config.parameters.denseParams = denseParams;

    return config;
  };

  auto gpuConfig = makeNoReluConfig(CNN::DeviceType::GPU);
  auto cpuConfig = makeNoReluConfig(CNN::DeviceType::CPU);

  CNN::Samples<float> samples(3);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f, 0.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[1].input.data = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
  samples[1].output = {0.0f, 1.0f};
  samples[2].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[2].input.data = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  samples[2].output = {1.0f, 0.0f};

  auto gpuCore = CNN::Core<float>::makeCore(gpuConfig);
  auto cpuCore = CNN::Core<float>::makeCore(cpuConfig);

  gpuCore->train(samples.size(), CNN::makeSampleProvider(samples));
  cpuCore->train(samples.size(), CNN::makeSampleProvider(samples));

  const auto& gp = gpuCore->getParameters();
  const auto& cp = cpuCore->getParameters();
  float tol = 1e-4f;

  CHECK_NEAR(gp.normParams[0].gamma[0], cp.normParams[0].gamma[0], tol, "NoRelu GPU-CPU gamma");
  CHECK_NEAR(gp.normParams[0].beta[0], cp.normParams[0].beta[0], tol, "NoRelu GPU-CPU beta");

  for (size_t i = 0; i < 4; i++) {
    CHECK_NEAR(gp.convParams[0].filters[i], cp.convParams[0].filters[i], tol,
               "NoRelu GPU-CPU filt[" + std::to_string(i) + "]");
  }

  std::cout << "  GPU filt=[" << gp.convParams[0].filters[0] << "," << gp.convParams[0].filters[1] << ","
            << gp.convParams[0].filters[2] << "," << gp.convParams[0].filters[3] << "]"
            << " gamma=" << gp.normParams[0].gamma[0] << " beta=" << gp.normParams[0].beta[0] << std::endl;
  std::cout << "  CPU filt=[" << cp.convParams[0].filters[0] << "," << cp.convParams[0].filters[1] << ","
            << cp.convParams[0].filters[2] << "," << cp.convParams[0].filters[3] << "]"
            << " gamma=" << cp.normParams[0].gamma[0] << " beta=" << cp.normParams[0].beta[0] << std::endl;
}

//===================================================================================================================//

// Test 2: Verify batch stats differ from instance stats (proves cross-sample reduction)
static void testGPUTrueBNBatchVsInstanceStats()
{
  std::cout << "--- testGPUTrueBNBatchVsInstanceStats ---" << std::endl;

  // Train with BATCHNORM (cross-sample stats) using 3 diverse samples
  auto bnConfig = makeGPUTrueBNTestConfig(1, ANN::ActvFuncType::SIGMOID, CNN::DeviceType::GPU);
  bnConfig.costFunctionConfig.type = CNN::CostFunctionType::SQUARED_DIFFERENCE;
  bnConfig.trainingConfig.batchSize = 3;

  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}};
  denseParams.biases[1] = {0.0f};
  bnConfig.parameters.denseParams = denseParams;

  // Train with INSTANCENORM (per-sample stats) using same params
  auto inConfig = bnConfig;
  inConfig.layersConfig.cnnLayers[1].type = CNN::LayerType::INSTANCENORM;

  // Use 3 diverse samples so batch stats differ from per-sample stats
  CNN::Samples<float> samples(3);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f};

  samples[1].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[1].input.data = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
  samples[1].output = {0.0f};

  samples[2].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[2].input.data = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
  samples[2].output = {0.5f};

  auto bnCore = CNN::Core<float>::makeCore(bnConfig);
  auto inCore = CNN::Core<float>::makeCore(inConfig);

  bnCore->train(samples.size(), CNN::makeSampleProvider(samples));
  inCore->train(samples.size(), CNN::makeSampleProvider(samples));

  const auto& bnP = bnCore->getParameters();
  const auto& inP = inCore->getParameters();

  // Running stats MUST differ between BATCHNORM and INSTANCENORM
  // because BATCHNORM computes mean/var across N*H*W while INSTANCENORM uses H*W per sample
  bool runningMeanDiffers = std::fabs(bnP.normParams[0].runningMean[0] - inP.normParams[0].runningMean[0]) > 1e-6f;
  bool runningVarDiffers = std::fabs(bnP.normParams[0].runningVar[0] - inP.normParams[0].runningVar[0]) > 1e-6f;

  // At least one should differ (with diverse samples, both should differ)
  CHECK(runningMeanDiffers || runningVarDiffers, "True BN running stats differ from InstanceNorm");

  std::cout << "  BN runningMean=" << bnP.normParams[0].runningMean[0]
            << "  IN runningMean=" << inP.normParams[0].runningMean[0] << std::endl;
  std::cout << "  BN runningVar=" << bnP.normParams[0].runningVar[0]
            << "  IN runningVar=" << inP.normParams[0].runningVar[0] << std::endl;

  // Gamma/beta should also differ due to different gradient paths
  bool gammaDiffers = std::fabs(bnP.normParams[0].gamma[0] - inP.normParams[0].gamma[0]) > 1e-6f;
  bool betaDiffers = std::fabs(bnP.normParams[0].beta[0] - inP.normParams[0].beta[0]) > 1e-6f;
  CHECK(gammaDiffers || betaDiffers, "True BN gamma/beta differ from InstanceNorm");

  std::cout << "  BN gamma=" << bnP.normParams[0].gamma[0] << "  IN gamma=" << inP.normParams[0].gamma[0] << std::endl;
  std::cout << "  BN beta=" << bnP.normParams[0].beta[0] << "  IN beta=" << inP.normParams[0].beta[0] << std::endl;
}

//===================================================================================================================//

// Test 3: True BN GPU training converges on a simple classification task

void runGPUBatchNormTests()
{
  testGPUTrueBNvsCPUParity();
  testGPUTrueBNNoRelu();
  testGPUTrueBNBatchVsInstanceStats();
}
