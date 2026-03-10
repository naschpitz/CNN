#include "test_helpers.hpp"

//===================================================================================================================//

static void testGPUEndToEnd()
{
  std::cout << "--- testGPUEndToEnd (train + predict) ---" << std::endl;

  // Same architecture as CPU test:
  // 1x5x5 → Conv(1 filter 3x3 valid) → 1x3x3 → ReLU → Flatten(9) → Dense(1, sigmoid)
  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;

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

  // Pre-initialized conv parameters (same as CPU test)
  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;
  config.numGPUs = 1;

  // "bright" (gradient-fill) → 1, "dark" (all 0s) → 0
  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  // Retry up to 5 times to handle random ANN weight initialization
  std::unique_ptr<CNN::Core<float>> core;
  CNN::Output<float> pred0, pred1;
  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    core = CNN::Core<float>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > 0.7f && pred1[0] < 0.3f)
      converged = true;
  }

  CHECK(core != nullptr, "GPU core creation");
  CHECK(pred0.size() == 1, "GPU predict output size 0");
  CHECK(pred1.size() == 1, "GPU predict output size 1");
  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "GPU bright > 0.7 & dark < 0.3 after training (5 attempts)");

  // Test method
  CNN::TestResult<float> result = core->test(samples.size(), CNN::makeSampleProvider(samples));
  CHECK(result.numSamples == 2, "GPU test numSamples");
  CHECK(result.averageLoss >= 0.0f, "GPU test avgLoss non-negative");
  CHECK(result.averageLoss < 0.1f, "GPU test avgLoss reasonably small");
  std::cout << "  test avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

static void testGPUPredictOnly()
{
  std::cout << "--- testGPUPredictOnly ---" << std::endl;

  // Train on CPU, then verify GPU predict gives same results
  CNN::CoreConfig<float> cpuConfig;
  cpuConfig.modeType = CNN::ModeType::TRAIN;
  cpuConfig.deviceType = CNN::DeviceType::CPU;
  cpuConfig.inputShape = {1, 5, 5};
  cpuConfig.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  cpuConfig.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  cpuConfig.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  cpuConfig.parameters.convParams = {initConv};

  cpuConfig.trainingConfig.numEpochs = 50;
  cpuConfig.trainingConfig.learningRate = 0.5f;
  cpuConfig.progressReports = 0;

  auto cpuCore = CNN::Core<float>::makeCore(cpuConfig);

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  cpuCore->train(samples.size(), CNN::makeSampleProvider(samples));

  // Get trained parameters from CPU core
  const CNN::Parameters<float>& trainedParams = cpuCore->getParameters();

  // Create a GPU core in PREDICT mode with the trained parameters
  CNN::CoreConfig<float> gpuConfig;
  gpuConfig.modeType = CNN::ModeType::PREDICT;
  gpuConfig.deviceType = CNN::DeviceType::GPU;
  gpuConfig.inputShape = {1, 5, 5};
  gpuConfig.logLevel = CNN::LogLevel::ERROR;
  gpuConfig.layersConfig = cpuConfig.layersConfig;
  gpuConfig.parameters = trainedParams;

  auto gpuCore = CNN::Core<float>::makeCore(gpuConfig);

  CNN::Output<float> cpuPred0 = cpuCore->predict(samples[0].input);
  CNN::Output<float> gpuPred0 = gpuCore->predict(samples[0].input);
  CNN::Output<float> cpuPred1 = cpuCore->predict(samples[1].input);
  CNN::Output<float> gpuPred1 = gpuCore->predict(samples[1].input);

  std::cout << "  CPU bright=" << cpuPred0[0] << "  GPU bright=" << gpuPred0[0] << std::endl;
  std::cout << "  CPU dark=" << cpuPred1[0] << "  GPU dark=" << gpuPred1[0] << std::endl;

  // GPU and CPU should produce very close results (float precision)
  CHECK_NEAR(cpuPred0[0], gpuPred0[0], 0.01f, "GPU vs CPU bright prediction");
  CHECK_NEAR(cpuPred1[0], gpuPred1[0], 0.01f, "GPU vs CPU dark prediction");
}

//===================================================================================================================//

static void testGPUWithPoolLayer()
{
  std::cout << "--- testGPUWithPoolLayer (Conv→ReLU→Pool→Conv→ReLU→Flatten→Dense) ---" << std::endl;

  // Same architecture as testConvPoolConv but on GPU
  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 10, 10};
  config.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig pool1;
  pool1.type = CNN::LayerType::POOL;
  pool1.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};

  CNN::CNNLayerConfig conv2;
  conv2.type = CNN::LayerType::CONV;
  conv2.config = CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu2;
  relu2.type = CNN::LayerType::RELU;
  relu2.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {conv1, relu1, pool1, conv2, relu2, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv1;
  initConv1.numFilters = 2;
  initConv1.inputC = 1;
  initConv1.filterH = 3;
  initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1f);
  initConv1.biases.assign(2, 0.0f);

  CNN::ConvParameters<float> initConv2;
  initConv2.numFilters = 1;
  initConv2.inputC = 2;
  initConv2.filterH = 3;
  initConv2.filterW = 3;
  initConv2.filters.assign(1 * 2 * 3 * 3, 0.1f);
  initConv2.biases.assign(1, 0.0f);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;
  config.numGPUs = 1;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 10, 10});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 10, 10}, 0.0f);
  samples[1].output = {0.0f};

  // Retry up to 5 times to handle random ANN weight initialization
  CNN::Output<float> pred0, pred1;
  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "GPU pool bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testGPUMultiConvStack()
{
  std::cout << "--- testGPUMultiConvStack (Conv→ReLU→Conv→ReLU→Flatten→Dense) ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 8, 8};
  config.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig conv2;
  conv2.type = CNN::LayerType::CONV;
  conv2.config = CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu2;
  relu2.type = CNN::LayerType::RELU;
  relu2.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {conv1, relu1, conv2, relu2, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv1;
  initConv1.numFilters = 2;
  initConv1.inputC = 1;
  initConv1.filterH = 3;
  initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1f);
  initConv1.biases.assign(2, 0.0f);

  CNN::ConvParameters<float> initConv2;
  initConv2.numFilters = 1;
  initConv2.inputC = 2;
  initConv2.filterH = 3;
  initConv2.filterW = 3;
  initConv2.filters.assign(1 * 2 * 3 * 3, 0.1f);
  initConv2.biases.assign(1, 0.0f);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;
  config.numGPUs = 1;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 8, 8});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 8, 8}, 0.0f);
  samples[1].output = {0.0f};

  // Retry up to 5 times to handle random ANN weight initialization
  CNN::Output<float> pred0, pred1;
  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "GPU multi-conv bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testGPUShuffleSamples()
{
  std::cout << "--- testGPUShuffleSamples ---" << std::endl;

  // Verify GPU training works with both shuffle=true and shuffle=false
  auto makeConfig = [](bool shuffle) {
    CNN::CoreConfig<float> config;
    config.modeType = CNN::ModeType::TRAIN;
    config.deviceType = CNN::DeviceType::GPU;
    config.inputShape = {1, 5, 5};
    config.logLevel = CNN::LogLevel::ERROR;
    config.numGPUs = 1;

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

    CNN::ConvParameters<float> initConv;
    initConv.numFilters = 1;
    initConv.inputC = 1;
    initConv.filterH = 3;
    initConv.filterW = 3;
    initConv.filters.assign(9, 0.1f);
    initConv.biases.assign(1, 0.0f);
    config.parameters.convParams = {initConv};

    config.trainingConfig.numEpochs = 100;
    config.trainingConfig.learningRate = 0.5f;
    config.trainingConfig.shuffleSamples = shuffle;
    config.progressReports = 0;
    return config;
  };

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  bool shuffleConverged = false;

  for (int attempt = 0; attempt < 5 && !shuffleConverged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(makeConfig(true));
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    auto p0 = core->predict(samples[0].input);
    auto p1 = core->predict(samples[1].input);

    if (p0[0] > p1[0])
      shuffleConverged = true;
  }

  CHECK(shuffleConverged, "GPU CNN shuffle=true converged (5 attempts)");

  bool noShuffleConverged = false;

  for (int attempt = 0; attempt < 5 && !noShuffleConverged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(makeConfig(false));
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    auto p0 = core->predict(samples[0].input);
    auto p1 = core->predict(samples[1].input);

    if (p0[0] > p1[0])
      noShuffleConverged = true;
  }

  CHECK(noShuffleConverged, "GPU CNN shuffle=false converged (5 attempts)");

  std::cout << "  shuffle=true: " << shuffleConverged << "  shuffle=false: " << noShuffleConverged << std::endl;
}

//===================================================================================================================//

static void testGPUCrossEntropyTraining()
{
  std::cout << "--- testGPUCrossEntropyTraining (CNN) ---" << std::endl;

  // 1x5x5 → Conv → ReLU → Flatten → Dense(3, softmax) with cross-entropy on GPU
  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 1;

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
  config.layersConfig.denseLayers = {{3, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  config.parameters.convParams = {initConv};

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<float> samples(3);
  samples[0].input = makeGradientInput<float>({1, 5, 5}, 0.8f, 1.0f);
  samples[0].output = {1.0f, 0.0f, 0.0f};
  samples[1].input = makeGradientInput<float>({1, 5, 5}, 0.0f, 0.2f);
  samples[1].output = {0.0f, 1.0f, 0.0f};
  samples[2].input = makeGradientInput<float>({1, 5, 5}, 0.4f, 0.6f);
  samples[2].output = {0.0f, 0.0f, 1.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  CNN::TestResult<float> result = core->test(samples.size(), CNN::makeSampleProvider(samples));

  CHECK(result.averageLoss >= 0.0f, "GPU CNN CE: loss non-negative");
  CHECK(std::isfinite(result.averageLoss), "GPU CNN CE: loss is finite");
  CHECK(result.numSamples == 3, "GPU CNN CE: 3 samples");

  // Softmax outputs should sum to 1
  auto out0 = core->predict(samples[0].input);
  float sum0 = out0[0] + out0[1] + out0[2];
  CHECK_NEAR(sum0, 1.0f, 0.01f, "GPU CNN CE: softmax sums to 1");

  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == CNN::CostFunctionType::CROSS_ENTROPY, "GPU CNN CE: type preserved");

  std::cout << "  GPU CNN CE avgLoss=" << result.averageLoss << " accuracy=" << result.accuracy << "%" << std::endl;
}

//===================================================================================================================//

static void testGPUWeightedCrossEntropyTraining()
{
  std::cout << "--- testGPUWeightedCrossEntropyTraining (CNN) ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 1;

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
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  config.parameters.convParams = {initConv};

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {5.0f, 1.0f};
  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f, 0.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f, 1.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  CNN::TestResult<float> result = core->test(samples.size(), CNN::makeSampleProvider(samples));

  CHECK(result.averageLoss >= 0.0f, "GPU CNN weighted CE: loss non-negative");
  CHECK(std::isfinite(result.averageLoss), "GPU CNN weighted CE: loss is finite");

  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == CNN::CostFunctionType::CROSS_ENTROPY, "GPU CNN weighted CE: type preserved");
  CHECK(cfc.weights.size() == 2, "GPU CNN weighted CE: weights preserved");
  CHECK_NEAR(cfc.weights[0], 5.0f, 1e-5f, "GPU CNN weighted CE: weight[0] = 5.0");

  std::cout << "  GPU CNN weighted CE avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

static void testGPUMultiChannelInput()
{
  std::cout << "--- testGPUMultiChannelInput ---" << std::endl;

  // 3x6x6 → Conv(2,3x3) → ReLU → Flatten(32) → Dense(1,sigmoid)
  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {3, 6, 6};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 1;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {conv1, relu1, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 2;
  initConv.inputC = 3;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(2 * 3 * 3 * 3, 0.05f);
  initConv.biases.assign(2, 0.0f);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({3, 6, 6}, 0.3f, 1.0f);
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({3, 6, 6}, 0.0f);
  samples[1].output = {0.0f};

  CNN::Output<float> pred0, pred1;
  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  GPU pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "GPU multichannel bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testGPUParameterRoundTrip()
{
  std::cout << "--- testGPUParameterRoundTrip ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 1;

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

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  auto core = CNN::Core<float>::makeCore(config);

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& params = core->getParameters();
  CHECK(params.convParams.size() == 1, "GPU param roundtrip convParams count");
  CHECK(params.convParams[0].filters.size() == 9, "GPU param roundtrip filters size");
  CHECK(params.convParams[0].biases.size() == 1, "GPU param roundtrip biases size");

  bool filtersChanged = false;

  for (ulong i = 0; i < 9; i++) {
    if (std::fabs(params.convParams[0].filters[i] - 0.1f) > 1e-6f) {
      filtersChanged = true;
      break;
    }
  }

  CHECK(filtersChanged, "GPU param roundtrip: filters changed after training");

  // Create predict core with trained params
  CNN::CoreConfig<float> predictConfig;
  predictConfig.modeType = CNN::ModeType::PREDICT;
  predictConfig.deviceType = CNN::DeviceType::GPU;
  predictConfig.inputShape = {1, 5, 5};
  predictConfig.logLevel = CNN::LogLevel::ERROR;
  predictConfig.layersConfig = config.layersConfig;
  predictConfig.parameters = params;

  auto predictCore = CNN::Core<float>::makeCore(predictConfig);
  CHECK(predictCore != nullptr, "GPU param roundtrip predict core creation");

  CNN::Output<float> origPred = core->predict(samples[0].input);
  CNN::Output<float> newPred = predictCore->predict(samples[0].input);
  CHECK_NEAR(origPred[0], newPred[0], 0.01f, "GPU param roundtrip prediction match");
  std::cout << "  GPU original=" << origPred[0] << "  from_params=" << newPred[0] << std::endl;
}

//===================================================================================================================//

static void testGPUParametersDuringTraining()
{
  std::cout << "--- testGPUParametersDuringTraining ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 1;

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

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 10;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  auto core = CNN::Core<float>::makeCore(config);

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  bool paramsChecked = false;
  bool convFiltersNonEmpty = false;
  bool denseWeightsNonEmpty = false;
  bool denseBiasesNonEmpty = false;
  ulong lastEpoch = 0;

  core->setTrainingCallback([&](const CNN::TrainingProgress<float>& progress) {
    if (progress.currentEpoch > lastEpoch && lastEpoch > 0 && !paramsChecked) {
      const CNN::Parameters<float>& params = core->getParameters();
      convFiltersNonEmpty = !params.convParams.empty() && !params.convParams[0].filters.empty();
      denseWeightsNonEmpty = !params.denseParams.weights.empty();
      denseBiasesNonEmpty = !params.denseParams.biases.empty();
      paramsChecked = true;
    }

    lastEpoch = progress.currentEpoch;
  });

  core->train(samples.size(), CNN::makeSampleProvider(samples));

  CHECK(paramsChecked, "GPU epoch transition detected in callback");
  CHECK(convFiltersNonEmpty, "GPU convParams[0].filters non-empty during training");
  CHECK(denseWeightsNonEmpty, "GPU denseParams.weights non-empty during training");
  CHECK(denseBiasesNonEmpty, "GPU denseParams.biases non-empty during training");
}

//===================================================================================================================//

static void testGPUMultipleOutputNeurons()
{
  std::cout << "--- testGPUMultipleOutputNeurons ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 8, 8};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 1;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{3, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 2;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(2 * 9, 0.1f);
  initConv.biases.assign(2, 0.0f);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 8, 8});
  samples[0].output = {1.0f, 0.0f, 1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 8, 8}, 0.0f);
  samples[1].output = {0.0f, 1.0f, 0.0f};

  CNN::Output<float> pred0, pred1;
  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  CHECK(pred0.size() == 3, "GPU multi-output size");
  std::cout << "  GPU pred(bright)=[" << pred0[0] << "," << pred0[1] << "," << pred0[2] << "]" << std::endl;
  CHECK(converged, "GPU multi-output[0] bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testGPUCostFunctionConfigGetter()
{
  std::cout << "--- testGPUCostFunctionConfigGetter ---" << std::endl;

  CNN::CoreConfig<float> configDefault;
  configDefault.modeType = CNN::ModeType::PREDICT;
  configDefault.deviceType = CNN::DeviceType::GPU;
  configDefault.inputShape = {1, 5, 5};
  configDefault.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};
  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};
  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  configDefault.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  configDefault.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SIGMOID}};

  auto coreDefault = CNN::Core<float>::makeCore(configDefault);
  const auto& cfcDefault = coreDefault->getCostFunctionConfig();
  CHECK(cfcDefault.type == CNN::CostFunctionType::SQUARED_DIFFERENCE, "GPU default type is squaredDifference");
  CHECK(cfcDefault.weights.empty(), "GPU default weights is empty");

  // Weighted config
  CNN::CoreConfig<float> configWeighted = configDefault;
  configWeighted.costFunctionConfig.type = CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
  configWeighted.costFunctionConfig.weights = {5.0f, 0.2f};

  auto coreWeighted = CNN::Core<float>::makeCore(configWeighted);
  const auto& cfcWeighted = coreWeighted->getCostFunctionConfig();
  CHECK(cfcWeighted.type == CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE, "GPU weighted type");
  CHECK(cfcWeighted.weights.size() == 2, "GPU weighted weights size = 2");
  CHECK_NEAR(cfcWeighted.weights[0], 5.0f, 1e-5f, "GPU weight[0] = 5.0");
  CHECK_NEAR(cfcWeighted.weights[1], 0.2f, 1e-5f, "GPU weight[1] = 0.2");
}

//===================================================================================================================//

static void testGPUWeightedLossTraining()
{
  std::cout << "--- testGPUWeightedLossTraining ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 1;

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
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  config.parameters.convParams = {initConv};

  config.costFunctionConfig.type = CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
  config.costFunctionConfig.weights = {5.0f, 1.0f};
  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f, 0.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f, 1.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  CNN::TestResult<float> result = core->test(samples.size(), CNN::makeSampleProvider(samples));
  CHECK(result.numSamples == 2, "GPU weighted CNN: tested 2 samples");
  CHECK(result.averageLoss >= 0.0f, "GPU weighted CNN: loss non-negative");
  CHECK(std::isfinite(result.averageLoss), "GPU weighted CNN: loss is finite");

  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE, "GPU weighted CNN: type preserved");
  CHECK(cfc.weights.size() == 2, "GPU weighted CNN: weights preserved");

  std::cout << "  GPU weighted avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

//-- Multi-GPU Tests --//
//===================================================================================================================//

static void testMultiGPUEndToEnd()
{
  std::cout << "--- testMultiGPUEndToEnd ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 2;

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

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  CNN::Output<float> pred0, pred1;
  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  multi-GPU pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "multi-GPU CNN: bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testMultiGPUTestMethod()
{
  std::cout << "--- testMultiGPUTestMethod ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 2;

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

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  CNN::TestResult<float> result = core->test(samples.size(), CNN::makeSampleProvider(samples));
  CHECK(result.numSamples == 2, "multi-GPU CNN test: numSamples = 2");
  CHECK(result.averageLoss >= 0.0f, "multi-GPU CNN test: loss >= 0");
  CHECK(std::isfinite(result.averageLoss), "multi-GPU CNN test: loss finite");
  std::cout << "  multi-GPU CNN test avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

static void testMultiGPUMultiChannelInput()
{
  std::cout << "--- testMultiGPUMultiChannelInput ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {3, 6, 6};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 2;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {conv1, relu1, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 2;
  initConv.inputC = 3;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(2 * 3 * 3 * 3, 0.05f);
  initConv.biases.assign(2, 0.0f);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({3, 6, 6}, 0.3f, 1.0f);
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({3, 6, 6}, 0.0f);
  samples[1].output = {0.0f};

  CNN::Output<float> pred0, pred1;
  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  multi-GPU multichannel pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "multi-GPU multichannel bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testMultiGPUCrossEntropyTraining()
{
  std::cout << "--- testMultiGPUCrossEntropyTraining ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 2;

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
  config.layersConfig.denseLayers = {{3, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  config.parameters.convParams = {initConv};

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<float> samples(3);
  samples[0].input = makeGradientInput<float>({1, 5, 5}, 0.8f, 1.0f);
  samples[0].output = {1.0f, 0.0f, 0.0f};
  samples[1].input = makeGradientInput<float>({1, 5, 5}, 0.0f, 0.2f);
  samples[1].output = {0.0f, 1.0f, 0.0f};
  samples[2].input = makeGradientInput<float>({1, 5, 5}, 0.4f, 0.6f);
  samples[2].output = {0.0f, 0.0f, 1.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  CNN::TestResult<float> result = core->test(samples.size(), CNN::makeSampleProvider(samples));

  CHECK(result.averageLoss >= 0.0f, "multi-GPU CNN CE: loss non-negative");
  CHECK(std::isfinite(result.averageLoss), "multi-GPU CNN CE: loss is finite");

  auto out0 = core->predict(samples[0].input);
  float sum0 = out0[0] + out0[1] + out0[2];
  CHECK_NEAR(sum0, 1.0f, 0.01f, "multi-GPU CNN CE: softmax sums to 1");

  std::cout << "  multi-GPU CNN CE avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

static void testMultiGPUParameterRoundTrip()
{
  std::cout << "--- testMultiGPUParameterRoundTrip ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 2;

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

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  auto core = CNN::Core<float>::makeCore(config);

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& params = core->getParameters();
  CHECK(params.convParams.size() == 1, "multi-GPU param roundtrip: convParams count");
  CHECK(params.convParams[0].filters.size() == 9, "multi-GPU param roundtrip: filters size");

  // Create predict core with trained params
  CNN::CoreConfig<float> predictConfig;
  predictConfig.modeType = CNN::ModeType::PREDICT;
  predictConfig.deviceType = CNN::DeviceType::GPU;
  predictConfig.inputShape = {1, 5, 5};
  predictConfig.logLevel = CNN::LogLevel::ERROR;
  predictConfig.layersConfig = config.layersConfig;
  predictConfig.parameters = params;

  auto predictCore = CNN::Core<float>::makeCore(predictConfig);
  CNN::Output<float> origPred = core->predict(samples[0].input);
  CNN::Output<float> newPred = predictCore->predict(samples[0].input);
  CHECK_NEAR(origPred[0], newPred[0], 0.01f, "multi-GPU param roundtrip: prediction match");
  std::cout << "  multi-GPU original=" << origPred[0] << "  from_params=" << newPred[0] << std::endl;
}

//===================================================================================================================//

static void testMultiGPUWithPoolLayer()
{
  std::cout << "--- testMultiGPUWithPoolLayer ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 8, 8};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 2;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig pool1;
  pool1.type = CNN::LayerType::POOL;
  pool1.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};

  CNN::CNNLayerConfig conv2;
  conv2.type = CNN::LayerType::CONV;
  conv2.config = CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu2;
  relu2.type = CNN::LayerType::RELU;
  relu2.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {conv1, relu1, pool1, conv2, relu2, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv1;
  initConv1.numFilters = 2;
  initConv1.inputC = 1;
  initConv1.filterH = 3;
  initConv1.filterW = 3;
  initConv1.filters.assign(2 * 9, 0.1f);
  initConv1.biases.assign(2, 0.0f);

  CNN::ConvParameters<float> initConv2;
  initConv2.numFilters = 1;
  initConv2.inputC = 2;
  initConv2.filterH = 3;
  initConv2.filterW = 3;
  initConv2.filters.assign(1 * 2 * 9, 0.1f);
  initConv2.biases.assign(1, 0.0f);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 8, 8});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 8, 8}, 0.0f);
  samples[1].output = {0.0f};

  CNN::Output<float> pred0, pred1;
  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  multi-GPU pool pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "multi-GPU with pool: bright > dark (5 attempts)");
}

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

void runGPUTests()
{
  // Single-GPU tests
  testGPUEndToEnd();
  testGPUPredictOnly();
  testGPUWithPoolLayer();
  testGPUMultiConvStack();
  testGPUShuffleSamples();
  testGPUCrossEntropyTraining();
  testGPUWeightedCrossEntropyTraining();
  testGPUMultiChannelInput();
  testGPUParameterRoundTrip();
  testGPUParametersDuringTraining();
  testGPUMultipleOutputNeurons();
  testGPUCostFunctionConfigGetter();
  testGPUWeightedLossTraining();
  testGPUExactForwardBackwardCrossEntropy();

  // Multi-GPU tests
  testMultiGPUEndToEnd();
  testMultiGPUTestMethod();
  testMultiGPUMultiChannelInput();
  testMultiGPUCrossEntropyTraining();
  testMultiGPUParameterRoundTrip();
  testMultiGPUWithPoolLayer();
}