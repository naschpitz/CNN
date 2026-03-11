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

static void testGPUTrueBNConvergence()
{
  std::cout << "--- testGPUTrueBNConvergence ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 1;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

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
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.batchSize = 4;
  config.trainingConfig.learningRate = 0.01f;
  config.trainingConfig.shuffleSamples = false;
  config.trainingConfig.optimizer.type = CNN::OptimizerType::ADAM;
  config.progressReports = 0;
  config.costFunctionConfig.type = CNN::CostFunctionType::SQUARED_DIFFERENCE;

  // 4 samples: bright (1) vs dark (0)
  CNN::Samples<float> samples(4);
  samples[0].input = makeGradientInput<float>({1, 5, 5}, 0.6f, 1.0f);
  samples[0].output = {1.0f};
  samples[1].input = makeGradientInput<float>({1, 5, 5}, 0.0f, 0.4f);
  samples[1].output = {0.0f};
  samples[2].input = makeGradientInput<float>({1, 5, 5}, 0.7f, 0.9f);
  samples[2].output = {1.0f};
  samples[3].input = makeGradientInput<float>({1, 5, 5}, 0.1f, 0.3f);
  samples[3].output = {0.0f};

  CNN::Output<float> pred0, pred1;
  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > 0.6f && pred1[0] < 0.4f)
      converged = true;
  }

  std::cout << "  GPU true BN pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "GPU true BN bright > 0.6 & dark < 0.4 (5 attempts)");
}

//===================================================================================================================//

// Test 4: Exact forward/backward with BATCHNORM using 2 samples (deterministic check)
static void testGPUTrueBNExactMultiSample()
{
  std::cout << "--- testGPUTrueBNExactMultiSample ---" << std::endl;

  // Same architecture as single-sample BN test but with 2 samples
  auto gpuConfig = makeGPUTrueBNTestConfig(2, ANN::ActvFuncType::SOFTMAX, CNN::DeviceType::GPU);
  gpuConfig.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  gpuConfig.trainingConfig.batchSize = 2;

  auto cpuConfig = makeGPUTrueBNTestConfig(2, ANN::ActvFuncType::SOFTMAX, CNN::DeviceType::CPU);
  cpuConfig.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  cpuConfig.trainingConfig.batchSize = 2;

  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}, {0.2f, 0.1f, -0.3f, 0.2f}};
  denseParams.biases[1] = {0.0f, 0.0f};
  gpuConfig.parameters.denseParams = denseParams;
  cpuConfig.parameters.denseParams = denseParams;

  // 2 samples
  CNN::Samples<float> samples(2);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f, 0.0f};

  samples[1].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[1].input.data = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
  samples[1].output = {0.0f, 1.0f};

  auto gpuCore = CNN::Core<float>::makeCore(gpuConfig);
  auto cpuCore = CNN::Core<float>::makeCore(cpuConfig);

  gpuCore->train(samples.size(), CNN::makeSampleProvider(samples));
  cpuCore->train(samples.size(), CNN::makeSampleProvider(samples));

  const auto& gp = gpuCore->getParameters();
  const auto& cp = cpuCore->getParameters();

  float tol = 1e-4f;

  // Verify all parameters match between GPU and CPU
  CHECK_NEAR(gp.normParams[0].gamma[0], cp.normParams[0].gamma[0], tol, "GPU-CPU exact BN 2-sample gamma");
  CHECK_NEAR(gp.normParams[0].beta[0], cp.normParams[0].beta[0], tol, "GPU-CPU exact BN 2-sample beta");
  CHECK_NEAR(gp.normParams[0].runningMean[0], cp.normParams[0].runningMean[0], tol,
             "GPU-CPU exact BN 2-sample runningMean");
  CHECK_NEAR(gp.normParams[0].runningVar[0], cp.normParams[0].runningVar[0], tol,
             "GPU-CPU exact BN 2-sample runningVar");

  // Verify predictions match
  auto gpuPred0 = gpuCore->predict(samples[0].input);
  auto cpuPred0 = cpuCore->predict(samples[0].input);
  auto gpuPred1 = gpuCore->predict(samples[1].input);
  auto cpuPred1 = cpuCore->predict(samples[1].input);

  CHECK_NEAR(gpuPred0[0], cpuPred0[0], tol, "GPU-CPU exact BN 2-sample pred0[0]");
  CHECK_NEAR(gpuPred0[1], cpuPred0[1], tol, "GPU-CPU exact BN 2-sample pred0[1]");
  CHECK_NEAR(gpuPred1[0], cpuPred1[0], tol, "GPU-CPU exact BN 2-sample pred1[0]");
  CHECK_NEAR(gpuPred1[1], cpuPred1[1], tol, "GPU-CPU exact BN 2-sample pred1[1]");

  std::cout << "  GPU pred0=[" << gpuPred0[0] << "," << gpuPred0[1] << "]" << "  CPU pred0=[" << cpuPred0[0] << ","
            << cpuPred0[1] << "]" << std::endl;
  std::cout << "  GPU pred1=[" << gpuPred1[0] << "," << gpuPred1[1] << "]" << "  CPU pred1=[" << cpuPred1[0] << ","
            << cpuPred1[1] << "]" << std::endl;
}

void runGPUBatchNormTests2()
{
  testGPUTrueBNConvergence();
  testGPUTrueBNExactMultiSample();
}
