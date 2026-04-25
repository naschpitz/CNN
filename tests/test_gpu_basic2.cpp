#include "test_helpers.hpp"

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
  config.trainingConfig.shuffleSeed = 42; // Fully deterministic — no retry loop.
  config.progressReports = 0;
  config.numGPUs = 1;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 8, 8});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 8, 8}, 0.0f);
  samples[1].output = {0.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));
  CNN::Output<float> pred0 = core->predict(samples[0].input).output;
  CNN::Output<float> pred1 = core->predict(samples[1].input).output;

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(pred0[0] > pred1[0], "GPU multi-conv bright > dark");
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
    auto p0 = core->predict(samples[0].input).output;
    auto p1 = core->predict(samples[1].input).output;

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
    auto p0 = core->predict(samples[0].input).output;
    auto p1 = core->predict(samples[1].input).output;

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
  auto out0 = core->predict(samples[0].input).output;
  float sum0 = out0[0] + out0[1] + out0[2];
  CHECK_NEAR(sum0, 1.0f, 0.01f, "GPU CNN CE: softmax sums to 1");

  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == CNN::CostFunctionType::CROSS_ENTROPY, "GPU CNN CE: type preserved");

  std::cout << "  GPU CNN CE avgLoss=" << result.averageLoss << " accuracy=" << result.accuracy << "%" << std::endl;
}

//===================================================================================================================//

void runGPUBasicTests2()
{
  testGPUMultiConvStack();
  testGPUShuffleSamples();
  testGPUCrossEntropyTraining();
}
