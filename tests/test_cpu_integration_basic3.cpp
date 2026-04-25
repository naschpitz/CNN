#include "test_helpers.hpp"

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
    auto p0 = core->predict(samples[0].input).output;
    auto p1 = core->predict(samples[1].input).output;

    if (p0[0] > p1[0])
      shuffleConverged = true;
  }

  CHECK(shuffleConverged, "CNN shuffle=true converged (5 attempts)");

  // Shuffle disabled
  bool noShuffleConverged = false;

  for (int attempt = 0; attempt < 5 && !noShuffleConverged; ++attempt) {
    auto core = CNN::Core<double>::makeCore(makeConfig(false));
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    auto p0 = core->predict(samples[0].input).output;
    auto p1 = core->predict(samples[1].input).output;

    if (p0[0] > p1[0])
      noShuffleConverged = true;
  }

  CHECK(noShuffleConverged, "CNN shuffle=false converged (5 attempts)");

  std::cout << "  shuffle=true: " << shuffleConverged << "  shuffle=false: " << noShuffleConverged << std::endl;
}

void runIntegrationBasicTests3()
{
  testCostFunctionConfigGetter();
  testWeightedLossTraining();
  testShuffleSamplesDefault();
  testShuffleSamplesTraining();
}
