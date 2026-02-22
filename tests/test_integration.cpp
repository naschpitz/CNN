#include "test_helpers.hpp"

//===================================================================================================================//

static void testEndToEnd() {
  std::cout << "--- testEndToEnd (train + predict) ---" << std::endl;

  // 1x5x5 → Conv(1 filter 3x3 valid) → 1x3x3 → ReLU → Flatten(9) → Dense(1, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.verbose = false;

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

  // Supply pre-initialized conv parameters to avoid dead-ReLU from random init.
  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1);  // All positive → sum = 0.9 > 0
  initConv.biases.assign(1, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  // "bright" (gradient-fill) → 1, "dark" (all 0s) → 0
  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  // Retry up to 5 times to handle random ANN weight initialization
  std::unique_ptr<CNN::Core<double>> core;
  CNN::Output<double> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    core = CNN::Core<double>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  CHECK(core != nullptr, "core creation");
  CHECK(pred0.size() == 1, "predict output size 0");
  CHECK(pred1.size() == 1, "predict output size 1");
  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "bright > dark after training (5 attempts)");

  // Test method
  CNN::TestResult<double> result = core->test(samples);
  CHECK(result.numSamples == 2, "test numSamples");
  CHECK(result.averageLoss >= 0.0, "test avgLoss non-negative");
  std::cout << "  test avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

static void testMultiConvStack() {
  std::cout << "--- testMultiConvStack (Conv→ReLU→Conv→ReLU→Flatten→Dense) ---" << std::endl;

  // 1x8x8 → Conv(2,3x3) → 2x6x6 → ReLU → Conv(1,3x3) → 1x4x4 → ReLU → Flatten(16) → Dense(1,sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 8, 8};
  config.verbose = false;

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

  // Pre-init conv params to avoid dead ReLU
  CNN::ConvParameters<double> initConv1;
  initConv1.numFilters = 2; initConv1.inputC = 1; initConv1.filterH = 3; initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1);
  initConv1.biases.assign(2, 0.0);

  CNN::ConvParameters<double> initConv2;
  initConv2.numFilters = 1; initConv2.inputC = 2; initConv2.filterH = 3; initConv2.filterW = 3;
  initConv2.filters.assign(1 * 2 * 3 * 3, 0.1);
  initConv2.biases.assign(1, 0.0);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 8, 8});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 8, 8}, 0.0);
  samples[1].output = {0.0};

  // Retry up to 5 times to handle random ANN weight initialization
  CNN::Output<double> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "multi-conv bright > dark (5 attempts)");
}


static void testConvPoolConv() {
  std::cout << "--- testConvPoolConv (Conv→ReLU→Pool→Conv→ReLU→Flatten→Dense) ---" << std::endl;

  // 1x10x10 → Conv(2,3x3) → 2x8x8 → ReLU → MaxPool(2x2,s2) → 2x4x4 → Conv(1,3x3) → 1x2x2 → ReLU → Flatten(4) → Dense(1,sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 10, 10};
  config.verbose = false;

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

  CNN::ConvParameters<double> initConv1;
  initConv1.numFilters = 2; initConv1.inputC = 1; initConv1.filterH = 3; initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1);
  initConv1.biases.assign(2, 0.0);

  CNN::ConvParameters<double> initConv2;
  initConv2.numFilters = 1; initConv2.inputC = 2; initConv2.filterH = 3; initConv2.filterW = 3;
  initConv2.filters.assign(1 * 2 * 3 * 3, 0.1);
  initConv2.biases.assign(1, 0.0);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 10, 10});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 10, 10}, 0.0);
  samples[1].output = {0.0};

  // Retry up to 5 times to handle random ANN weight initialization
  CNN::Output<double> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "conv-pool-conv bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testMultiChannelInput() {
  std::cout << "--- testMultiChannelInput (3-channel RGB-like) ---" << std::endl;

  // 3x6x6 → Conv(2,3x3) → 2x4x4 → ReLU → Flatten(32) → Dense(1,sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {3, 6, 6};
  config.verbose = false;

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

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 2; initConv.inputC = 3; initConv.filterH = 3; initConv.filterW = 3;
  initConv.filters.assign(2 * 3 * 3 * 3, 0.05);  // small positive values
  initConv.biases.assign(2, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({3, 6, 6}, 0.3, 1.0);
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({3, 6, 6}, 0.0);
  samples[1].output = {0.0};

  // Retry up to 5 times to handle random ANN weight initialization
  CNN::Output<double> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "multichannel bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testParameterRoundTrip() {
  std::cout << "--- testParameterRoundTrip ---" << std::endl;

  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.verbose = false;

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

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1; initConv.inputC = 1; initConv.filterH = 3; initConv.filterW = 3;
  initConv.filters.assign(9, 0.1);
  initConv.biases.assign(1, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  auto core = CNN::Core<double>::makeCore(config);

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  core->train(samples);

  // Get trained parameters
  const CNN::Parameters<double>& params = core->getParameters();
  CHECK(params.convParams.size() == 1, "param roundtrip convParams count");
  CHECK(params.convParams[0].filters.size() == 9, "param roundtrip filters size");
  CHECK(params.convParams[0].biases.size() == 1, "param roundtrip biases size");

  // Parameters should have changed from initial values after training
  bool filtersChanged = false;
  for (ulong i = 0; i < 9; i++) {
    if (std::fabs(params.convParams[0].filters[i] - 0.1) > 1e-6) {
      filtersChanged = true;
      break;
    }
  }
  CHECK(filtersChanged, "param roundtrip: filters changed after training");

  // Create a new core in PREDICT mode with trained parameters
  CNN::CoreConfig<double> predictConfig;
  predictConfig.modeType = CNN::ModeType::PREDICT;
  predictConfig.deviceType = CNN::DeviceType::CPU;
  predictConfig.inputShape = {1, 5, 5};
  predictConfig.verbose = false;
  predictConfig.layersConfig = config.layersConfig;
  predictConfig.parameters = params;

  auto predictCore = CNN::Core<double>::makeCore(predictConfig);
  CHECK(predictCore != nullptr, "param roundtrip predict core creation");

  // Predictions should match
  CNN::Output<double> origPred = core->predict(samples[0].input);
  CNN::Output<double> newPred = predictCore->predict(samples[0].input);
  CHECK_NEAR(origPred[0], newPred[0], 1e-9, "param roundtrip prediction match");
  std::cout << "  original=" << origPred[0] << "  from_params=" << newPred[0] << std::endl;
}

//===================================================================================================================//

static void testMultipleOutputNeurons() {
  std::cout << "--- testMultipleOutputNeurons ---" << std::endl;

  // 1x8x8 → Conv(2,3x3) → ReLU → Flatten(72) → Dense(3, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 8, 8};
  config.verbose = false;

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

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 2; initConv.inputC = 1; initConv.filterH = 3; initConv.filterW = 3;
  initConv.filters.assign(2 * 9, 0.1);
  initConv.biases.assign(2, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 8, 8});
  samples[0].output = {1.0, 0.0, 1.0};  // target: [1, 0, 1]
  samples[1].input = CNN::Tensor3D<double>({1, 8, 8}, 0.0);
  samples[1].output = {0.0, 1.0, 0.0};  // target: [0, 1, 0]

  // Retry up to 5 times to handle random ANN weight initialization
  CNN::Output<double> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  CHECK(pred0.size() == 3, "multi-output size");
  std::cout << "  pred(bright)=[" << pred0[0] << "," << pred0[1] << "," << pred0[2] << "]" << std::endl;
  CHECK(converged, "multi-output[0] bright > dark (5 attempts)");
}

//===================================================================================================================//

void runIntegrationTests() {
  testEndToEnd();
  testMultiConvStack();
  testConvPoolConv();
  testMultiChannelInput();
  testParameterRoundTrip();
  testMultipleOutputNeurons();
}