#include "test_helpers.hpp"

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
    pred0 = core->predict(samples[0].input).output;
    pred1 = core->predict(samples[1].input).output;

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

void runGPUBasicTests4()
{
  testGPUMultipleOutputNeurons();
  testGPUCostFunctionConfigGetter();
  testGPUWeightedLossTraining();
}
