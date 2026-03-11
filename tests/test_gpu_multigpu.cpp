#include "test_helpers.hpp"

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

void runGPUMultiGPUTests()
{
  testMultiGPUEndToEnd();
  testMultiGPUTestMethod();
  testMultiGPUMultiChannelInput();
}
