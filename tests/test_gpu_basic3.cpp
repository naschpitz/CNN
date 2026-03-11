#include "test_helpers.hpp"

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

void runGPUBasicTests3()
{
  testGPUWeightedCrossEntropyTraining();
  testGPUMultiChannelInput();
  testGPUParameterRoundTrip();
  testGPUParametersDuringTraining();
}
