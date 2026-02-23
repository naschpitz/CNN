#include "test_helpers.hpp"

//===================================================================================================================//

static void testGPUEndToEnd() {
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
  config.trainingConfig.numGPUs = 1;

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
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    core = CNN::Core<float>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > 0.7f && pred1[0] < 0.3f) converged = true;
  }

  CHECK(core != nullptr, "GPU core creation");
  CHECK(pred0.size() == 1, "GPU predict output size 0");
  CHECK(pred1.size() == 1, "GPU predict output size 1");
  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "GPU bright > 0.7 & dark < 0.3 after training (5 attempts)");

  // Test method
  CNN::TestResult<float> result = core->test(samples);
  CHECK(result.numSamples == 2, "GPU test numSamples");
  CHECK(result.averageLoss >= 0.0f, "GPU test avgLoss non-negative");
  CHECK(result.averageLoss < 0.1f, "GPU test avgLoss reasonably small");
  std::cout << "  test avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

static void testGPUPredictOnly() {
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

  cpuCore->train(samples);

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

static void testGPUWithPoolLayer() {
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
  initConv1.numFilters = 2; initConv1.inputC = 1; initConv1.filterH = 3; initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1f);
  initConv1.biases.assign(2, 0.0f);

  CNN::ConvParameters<float> initConv2;
  initConv2.numFilters = 1; initConv2.inputC = 2; initConv2.filterH = 3; initConv2.filterW = 3;
  initConv2.filters.assign(1 * 2 * 3 * 3, 0.1f);
  initConv2.biases.assign(1, 0.0f);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;
  config.trainingConfig.numGPUs = 1;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 10, 10});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 10, 10}, 0.0f);
  samples[1].output = {0.0f};

  // Retry up to 5 times to handle random ANN weight initialization
  CNN::Output<float> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "GPU pool bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testGPUMultiConvStack() {
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
  initConv1.numFilters = 2; initConv1.inputC = 1; initConv1.filterH = 3; initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1f);
  initConv1.biases.assign(2, 0.0f);

  CNN::ConvParameters<float> initConv2;
  initConv2.numFilters = 1; initConv2.inputC = 2; initConv2.filterH = 3; initConv2.filterW = 3;
  initConv2.filters.assign(1 * 2 * 3 * 3, 0.1f);
  initConv2.biases.assign(1, 0.0f);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;
  config.trainingConfig.numGPUs = 1;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 8, 8});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 8, 8}, 0.0f);
  samples[1].output = {0.0f};

  // Retry up to 5 times to handle random ANN weight initialization
  CNN::Output<float> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "GPU multi-conv bright > dark (5 attempts)");
}

//===================================================================================================================//

void runGPUTests() {
  testGPUEndToEnd();
  testGPUPredictOnly();
  testGPUWithPoolLayer();
  testGPUMultiConvStack();
}