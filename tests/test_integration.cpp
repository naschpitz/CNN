#include "test_helpers.hpp"

//===================================================================================================================//

static void testEndToEnd()
{
  std::cout << "--- testEndToEnd (train + predict) ---" << std::endl;

  // 1x5x5 → Conv(1 filter 3x3 valid) → 1x3x3 → ReLU → Flatten(9) → Dense(1, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
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

  // Supply pre-initialized conv parameters to avoid dead-ReLU from random init.
  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1); // All positive → sum = 0.9 > 0
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
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  CHECK(core != nullptr, "core creation");
  CHECK(pred0.size() == 1, "predict output size 0");
  CHECK(pred1.size() == 1, "predict output size 1");
  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "bright > dark after training (5 attempts)");

  // Test method
  CNN::TestResult<double> result = core->test(samples.size(), CNN::makeSampleProvider(samples));
  CHECK(result.numSamples == 2, "test numSamples");
  CHECK(result.averageLoss >= 0.0, "test avgLoss non-negative");
  CHECK(result.numCorrect <= result.numSamples, "numCorrect <= numSamples");
  CHECK(result.accuracy >= 0.0 && result.accuracy <= 100.0, "accuracy in [0, 100]");
  std::cout << "  test avgLoss=" << result.averageLoss << " accuracy=" << result.accuracy << "%" << std::endl;
}

//===================================================================================================================//

static void testMultiConvStack()
{
  std::cout << "--- testMultiConvStack (Conv→ReLU→Conv→ReLU→Flatten→Dense) ---" << std::endl;

  // 1x8x8 → Conv(2,3x3) → 2x6x6 → ReLU → Conv(1,3x3) → 1x4x4 → ReLU → Flatten(16) → Dense(1,sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
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

  // Pre-init conv params to avoid dead ReLU
  CNN::ConvParameters<double> initConv1;
  initConv1.numFilters = 2;
  initConv1.inputC = 1;
  initConv1.filterH = 3;
  initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1);
  initConv1.biases.assign(2, 0.0);

  CNN::ConvParameters<double> initConv2;
  initConv2.numFilters = 1;
  initConv2.inputC = 2;
  initConv2.filterH = 3;
  initConv2.filterW = 3;
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
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "multi-conv bright > dark (5 attempts)");
}

static void testConvPoolConv()
{
  std::cout << "--- testConvPoolConv (Conv→ReLU→Pool→Conv→ReLU→Flatten→Dense) ---" << std::endl;

  // 1x10x10 → Conv(2,3x3) → 2x8x8 → ReLU → MaxPool(2x2,s2) → 2x4x4 → Conv(1,3x3) → 1x2x2 → ReLU → Flatten(4) → Dense(1,sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
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

  CNN::ConvParameters<double> initConv1;
  initConv1.numFilters = 2;
  initConv1.inputC = 1;
  initConv1.filterH = 3;
  initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1);
  initConv1.biases.assign(2, 0.0);

  CNN::ConvParameters<double> initConv2;
  initConv2.numFilters = 1;
  initConv2.inputC = 2;
  initConv2.filterH = 3;
  initConv2.filterW = 3;
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
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "conv-pool-conv bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testMultiChannelInput()
{
  std::cout << "--- testMultiChannelInput (3-channel RGB-like) ---" << std::endl;

  // 3x6x6 → Conv(2,3x3) → 2x4x4 → ReLU → Flatten(32) → Dense(1,sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {3, 6, 6};
  config.logLevel = CNN::LogLevel::ERROR;

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
  initConv.numFilters = 2;
  initConv.inputC = 3;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(2 * 3 * 3 * 3, 0.05); // small positive values
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
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "multichannel bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testParameterRoundTrip()
{
  std::cout << "--- testParameterRoundTrip ---" << std::endl;

  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
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

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
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

  core->train(samples.size(), CNN::makeSampleProvider(samples));

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
  predictConfig.logLevel = CNN::LogLevel::ERROR;
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

static void testParametersDuringTraining()
{
  std::cout << "--- testParametersDuringTraining ---" << std::endl;

  // Train and verify getParameters() returns populated conv AND dense params
  // during training (in the callback), not just after training ends.
  // This test would FAIL without the dense params sync fix in CNN_CoreCPU.cpp.
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
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

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1);
  initConv.biases.assign(1, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 10;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  auto core = CNN::Core<double>::makeCore(config);

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  bool paramsChecked = false;
  bool convFiltersNonEmpty = false;
  bool denseWeightsNonEmpty = false;
  bool denseBiasesNonEmpty = false;
  ulong lastEpoch = 0;

  core->setTrainingCallback([&](const CNN::TrainingProgress<double>& progress) {
    // Detect epoch transition (first callback of a new epoch)
    if (progress.currentEpoch > lastEpoch && lastEpoch > 0 && !paramsChecked) {
      const CNN::Parameters<double>& params = core->getParameters();
      convFiltersNonEmpty = !params.convParams.empty() && !params.convParams[0].filters.empty();
      denseWeightsNonEmpty = !params.denseParams.weights.empty();
      denseBiasesNonEmpty = !params.denseParams.biases.empty();
      paramsChecked = true;
    }

    lastEpoch = progress.currentEpoch;
  });

  core->train(samples.size(), CNN::makeSampleProvider(samples));

  CHECK(paramsChecked, "epoch transition detected in callback");
  CHECK(convFiltersNonEmpty, "convParams[0].filters non-empty during training");
  CHECK(denseWeightsNonEmpty, "denseParams.weights non-empty during training");
  CHECK(denseBiasesNonEmpty, "denseParams.biases non-empty during training");
}

//===================================================================================================================//

static void testMultipleOutputNeurons()
{
  std::cout << "--- testMultipleOutputNeurons ---" << std::endl;

  // 1x8x8 → Conv(2,3x3) → ReLU → Flatten(72) → Dense(3, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 8, 8};
  config.logLevel = CNN::LogLevel::ERROR;

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
  initConv.numFilters = 2;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(2 * 9, 0.1);
  initConv.biases.assign(2, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 8, 8});
  samples[0].output = {1.0, 0.0, 1.0}; // target: [1, 0, 1]
  samples[1].input = CNN::Tensor3D<double>({1, 8, 8}, 0.0);
  samples[1].output = {0.0, 1.0, 0.0}; // target: [0, 1, 0]

  // Retry up to 5 times to handle random ANN weight initialization
  CNN::Output<double> pred0, pred1;
  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  CHECK(pred0.size() == 3, "multi-output size");
  std::cout << "  pred(bright)=[" << pred0[0] << "," << pred0[1] << "," << pred0[2] << "]" << std::endl;
  CHECK(converged, "multi-output[0] bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testCostFunctionConfigGetter()
{
  std::cout << "--- testCostFunctionConfigGetter ---" << std::endl;

  // Default config
  CNN::CoreConfig<double> configDefault;
  configDefault.modeType = CNN::ModeType::PREDICT;
  configDefault.deviceType = CNN::DeviceType::CPU;
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

  auto coreDefault = CNN::Core<double>::makeCore(configDefault);
  const auto& cfcDefault = coreDefault->getCostFunctionConfig();
  CHECK(cfcDefault.type == CNN::CostFunctionType::SQUARED_DIFFERENCE, "CNN default type is squaredDifference");
  CHECK(cfcDefault.weights.empty(), "CNN default weights is empty");

  // Weighted config
  CNN::CoreConfig<double> configWeighted = configDefault;
  configWeighted.costFunctionConfig.type = CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
  configWeighted.costFunctionConfig.weights = {5.0, 0.2};

  auto coreWeighted = CNN::Core<double>::makeCore(configWeighted);
  const auto& cfcWeighted = coreWeighted->getCostFunctionConfig();
  CHECK(cfcWeighted.type == CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE, "CNN weighted type");
  CHECK(cfcWeighted.weights.size() == 2, "CNN weighted weights size = 2");
  CHECK_NEAR(cfcWeighted.weights[0], 5.0, 1e-10, "CNN weight[0] = 5.0");
  CHECK_NEAR(cfcWeighted.weights[1], 0.2, 1e-10, "CNN weight[1] = 0.2");
}

//===================================================================================================================//

static void testWeightedLossTraining()
{
  std::cout << "--- testWeightedLossTraining ---" << std::endl;

  // 1x5x5 → Conv(1,3x3) → ReLU → Flatten(9) → Dense(2, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
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
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1);
  initConv.biases.assign(1, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  // Apply weighted loss
  config.costFunctionConfig.type = CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
  config.costFunctionConfig.weights = {5.0, 1.0};

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0, 0.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0, 1.0};

  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  // Verify training completed and test produces valid results
  CNN::TestResult<double> result = core->test(samples.size(), CNN::makeSampleProvider(samples));
  CHECK(result.numSamples == 2, "weighted CNN: tested 2 samples");
  CHECK(result.averageLoss >= 0.0, "weighted CNN: loss non-negative");
  CHECK(std::isfinite(result.averageLoss), "weighted CNN: loss is finite");
  CHECK(result.numCorrect <= 2, "weighted CNN: numCorrect <= 2");

  // Verify getter returns correct config after training
  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE, "weighted CNN: type preserved");
  CHECK(cfc.weights.size() == 2, "weighted CNN: weights preserved");

  std::cout << "  weighted avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

static void testShuffleSamplesDefault()
{
  std::cout << "--- testShuffleSamplesDefault ---" << std::endl;

  CNN::TrainingConfig<double> tc;
  CHECK(tc.shuffleSamples == true, "CNN shuffleSamples default is true");
}

//===================================================================================================================//

static void testShuffleSamplesTraining()
{
  std::cout << "--- testShuffleSamplesTraining ---" << std::endl;

  // Train with shuffle=true and shuffle=false, both should converge
  auto makeConfig = [](bool shuffle) {
    CNN::CoreConfig<double> config;
    config.modeType = CNN::ModeType::TRAIN;
    config.deviceType = CNN::DeviceType::CPU;
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

    CNN::ConvParameters<double> initConv;
    initConv.numFilters = 1;
    initConv.inputC = 1;
    initConv.filterH = 3;
    initConv.filterW = 3;
    initConv.filters.assign(9, 0.1);
    initConv.biases.assign(1, 0.0);
    config.parameters.convParams = {initConv};

    config.trainingConfig.numEpochs = 100;
    config.trainingConfig.learningRate = 0.5f;
    config.trainingConfig.shuffleSamples = shuffle;
    config.progressReports = 0;
    return config;
  };

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  // Shuffle enabled
  bool shuffleConverged = false;

  for (int attempt = 0; attempt < 5 && !shuffleConverged; ++attempt) {
    auto core = CNN::Core<double>::makeCore(makeConfig(true));
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    auto p0 = core->predict(samples[0].input);
    auto p1 = core->predict(samples[1].input);

    if (p0[0] > p1[0])
      shuffleConverged = true;
  }

  CHECK(shuffleConverged, "CNN shuffle=true converged (5 attempts)");

  // Shuffle disabled
  bool noShuffleConverged = false;

  for (int attempt = 0; attempt < 5 && !noShuffleConverged; ++attempt) {
    auto core = CNN::Core<double>::makeCore(makeConfig(false));
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    auto p0 = core->predict(samples[0].input);
    auto p1 = core->predict(samples[1].input);

    if (p0[0] > p1[0])
      noShuffleConverged = true;
  }

  CHECK(noShuffleConverged, "CNN shuffle=false converged (5 attempts)");

  std::cout << "  shuffle=true: " << shuffleConverged << "  shuffle=false: " << noShuffleConverged << std::endl;
}

//===================================================================================================================//

static void testCrossEntropyGradientNumerical()
{
  std::cout << "--- testCrossEntropyGradientNumerical ---" << std::endl;

  // Verify the analytical gradient formula matches numerical finite differences.
  // Loss = -sum(w_i * y_i * log(a_i))
  // dL/da_j = -w_j * y_j / a_j

  std::vector<double> activations = {0.7, 0.2, 0.1};
  std::vector<double> targets = {1.0, 0.0, 0.0};
  std::vector<double> weights = {1.0, 1.0, 1.0};

  auto computeLoss = [&](const std::vector<double>& a) {
    double loss = 0;
    const double epsilon = 1e-7;

    for (ulong i = 0; i < a.size(); i++) {
      double pred = std::max(a[i], epsilon);
      loss -= weights[i] * targets[i] * std::log(pred);
    }

    return loss;
  };

  double eps = 1e-6;

  for (ulong j = 0; j < activations.size(); j++) {
    const double epsilon = 1e-7;
    double pred = std::max(activations[j], epsilon);
    double analyticalGrad = -weights[j] * targets[j] / pred;

    std::vector<double> aPlus = activations;
    aPlus[j] += eps;
    std::vector<double> aMinus = activations;
    aMinus[j] -= eps;
    double numericalGrad = (computeLoss(aPlus) - computeLoss(aMinus)) / (2.0 * eps);

    std::cout << "  j=" << j << " analytical=" << analyticalGrad << " numerical=" << numericalGrad << std::endl;
    CHECK_NEAR(analyticalGrad, numericalGrad, 1e-4, "CE gradient numerical check");
  }

  // Also test with non-uniform weights
  weights = {5.0, 1.0, 2.0};
  targets = {0.0, 1.0, 0.0};
  activations = {0.1, 0.6, 0.3};

  for (ulong j = 0; j < activations.size(); j++) {
    const double epsilon = 1e-7;
    double pred = std::max(activations[j], epsilon);
    double analyticalGrad = -weights[j] * targets[j] / pred;

    std::vector<double> aPlus = activations;
    aPlus[j] += eps;
    std::vector<double> aMinus = activations;
    aMinus[j] -= eps;
    double numericalGrad = (computeLoss(aPlus) - computeLoss(aMinus)) / (2.0 * eps);

    std::cout << "  weighted j=" << j << " analytical=" << analyticalGrad << " numerical=" << numericalGrad
              << std::endl;
    CHECK_NEAR(analyticalGrad, numericalGrad, 1e-4, "weighted CE gradient numerical check");
  }
}

//===================================================================================================================//

static void testCrossEntropyLossDecreases()
{
  std::cout << "--- testCrossEntropyLossDecreases (CNN) ---" << std::endl;

  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;

  // 2 filters with varied init to break symmetry
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
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 2;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters = {
    0.1,  -0.2, 0.3,  -0.1, 0.4,  -0.3, 0.2,  -0.1, 0.5, // filter 0
    -0.3, 0.1,  -0.2, 0.4,  -0.1, 0.2,  -0.4, 0.3,  -0.1 // filter 1
  };

  initConv.biases.assign(2, 0.0);
  config.parameters.convParams = {initConv};

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;

  // Both samples have non-zero inputs so conv produces meaningful features
  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5}, 0.6, 1.0);
  samples[0].output = {1.0, 0.0};
  samples[1].input = makeGradientInput<double>({1, 5, 5}, 0.0, 0.4);
  samples[1].output = {0.0, 1.0};

  // Train 50 epochs, measure loss
  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));
  CNN::TestResult<double> result1 = core->test(samples.size(), CNN::makeSampleProvider(samples));

  // Train 200 more epochs from same params
  config.trainingConfig.numEpochs = 200;
  config.parameters = core->getParameters();
  auto core2 = CNN::Core<double>::makeCore(config);
  core2->train(samples.size(), CNN::makeSampleProvider(samples));
  CNN::TestResult<double> result2 = core2->test(samples.size(), CNN::makeSampleProvider(samples));

  std::cout << "  loss after 50=" << result1.averageLoss << "  after 250=" << result2.averageLoss << std::endl;
  CHECK(result2.averageLoss < result1.averageLoss, "CNN CE loss decreases with more training");
  CHECK(result2.averageLoss < 0.5, "CNN CE loss below 0.5 after 250 epochs on trivial problem");
}

//===================================================================================================================//

static void testCrossEntropyTraining()
{
  std::cout << "--- testCrossEntropyTraining (CNN) ---" << std::endl;

  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;

  // Use 4 filters with varied init so flatten features are diverse enough for 3-class softmax
  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{3, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 4;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  // Varied init: break symmetry so each filter extracts different features
  initConv.filters = {
    0.1,  -0.2, 0.3,  -0.1, 0.4,  -0.3, 0.2,  -0.1, 0.5, // filter 0
    -0.3, 0.1,  -0.2, 0.4,  -0.1, 0.2,  -0.4, 0.3,  -0.1, // filter 1
    0.2,  0.3,  -0.1, -0.2, 0.1,  0.4,  -0.3, -0.1, 0.2, // filter 2
    -0.1, -0.3, 0.2,  0.1,  -0.4, 0.3,  0.2,  -0.2, -0.1 // filter 3
  };

  initConv.biases.assign(4, 0.0);
  config.parameters.convParams = {initConv};

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;

  // 3-class classification with clearly different inputs
  CNN::Samples<double> samples(3);
  samples[0].input = makeGradientInput<double>({1, 5, 5}, 0.8, 1.0);
  samples[0].output = {1.0, 0.0, 0.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0, 1.0, 0.0};
  samples[2].input = makeGradientInput<double>({1, 5, 5}, 0.3, 0.5);
  samples[2].output = {0.0, 0.0, 1.0};

  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;

    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));

    CNN::TestResult<double> result = core->test(samples.size(), CNN::makeSampleProvider(samples));

    auto out0 = core->predict(samples[0].input);
    double sum0 = out0[0] + out0[1] + out0[2];

    bool lossOk = result.averageLoss < 1.0 && std::isfinite(result.averageLoss);
    bool softmaxOk = std::fabs(sum0 - 1.0) < 1e-5;

    if (lossOk && softmaxOk) {
      converged = true;
      CHECK(result.numSamples == 3, "CNN CE: 3 samples");

      const auto& cfc = core->getCostFunctionConfig();
      CHECK(cfc.type == CNN::CostFunctionType::CROSS_ENTROPY, "CNN CE: type preserved");

      std::cout << "  CNN CE avgLoss=" << result.averageLoss << " accuracy=" << result.accuracy << "%" << std::endl;
    }
  }

  CHECK(converged, "CNN CE: converged (loss < 1.0) in 5 attempts");
}

//===================================================================================================================//

static void testWeightedCrossEntropyTraining()
{
  std::cout << "--- testWeightedCrossEntropyTraining (CNN) ---" << std::endl;

  // Cross-entropy with per-class weights
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
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
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1);
  initConv.biases.assign(1, 0.0);
  config.parameters.convParams = {initConv};

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {5.0, 1.0};
  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0, 0.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0, 1.0};

  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  CNN::TestResult<double> result = core->test(samples.size(), CNN::makeSampleProvider(samples));

  CHECK(result.averageLoss >= 0.0, "CNN weighted CE: loss non-negative");
  CHECK(std::isfinite(result.averageLoss), "CNN weighted CE: loss is finite");

  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == CNN::CostFunctionType::CROSS_ENTROPY, "CNN weighted CE: type preserved");
  CHECK(cfc.weights.size() == 2, "CNN weighted CE: weights preserved");
  CHECK_NEAR(cfc.weights[0], 5.0, 1e-10, "CNN weighted CE: weight[0] = 5.0");

  std::cout << "  CNN weighted CE avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

void runIntegrationTests()
{
  testEndToEnd();
  testMultiConvStack();
  testConvPoolConv();
  testMultiChannelInput();
  testParameterRoundTrip();
  testParametersDuringTraining();
  testMultipleOutputNeurons();
  testCostFunctionConfigGetter();
  testWeightedLossTraining();
  testCrossEntropyGradientNumerical();
  testCrossEntropyLossDecreases();
  testCrossEntropyTraining();
  testWeightedCrossEntropyTraining();
  testShuffleSamplesDefault();
  testShuffleSamplesTraining();
}