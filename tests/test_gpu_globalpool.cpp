#include "test_helpers.hpp"

//===================================================================================================================//

static void testGPUWithGlobalAvgPool()
{
  std::cout << "--- testGPUWithGlobalAvgPool (Conv→ReLU→GAP→Flatten→Dense) ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig gapLayer;
  gapLayer.type = CNN::LayerType::GLOBALAVGPOOL;
  gapLayer.config = CNN::GlobalAvgPoolLayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {conv1, relu1, gapLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv1;
  initConv1.numFilters = 2;
  initConv1.inputC = 1;
  initConv1.filterH = 3;
  initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1f);
  initConv1.biases.assign(2, 0.0f);

  config.parameters.convParams = {initConv1};
  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.shuffleSeed = 42; // Fully deterministic — no retry loop.
  config.progressReports = 0;
  config.numGPUs = 1;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));
  CNN::Output<float> pred0 = core->predict(samples[0].input).output;
  CNN::Output<float> pred1 = core->predict(samples[1].input).output;

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(pred0[0] > pred1[0], "GPU gap bright > dark");
}

//===================================================================================================================//

static void testGPUWithGlobalDualPool()
{
  std::cout << "--- testGPUWithGlobalDualPool (Conv→ReLU→GDP→Flatten→Dense) ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig gdpLayer;
  gdpLayer.type = CNN::LayerType::GLOBALDUALPOOL;
  gdpLayer.config = CNN::GlobalDualPoolLayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  // 2 conv filters -> GDP outputs 4 features (2 avg + 2 max) -> dense 1
  config.layersConfig.cnnLayers = {conv1, relu1, gdpLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv1;
  initConv1.numFilters = 2;
  initConv1.inputC = 1;
  initConv1.filterH = 3;
  initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1f);
  initConv1.biases.assign(2, 0.0f);

  config.parameters.convParams = {initConv1};
  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.shuffleSeed = 42; // Fully deterministic — no retry loop.
  config.progressReports = 0;
  config.numGPUs = 1;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));
  CNN::Output<float> pred0 = core->predict(samples[0].input).output;
  CNN::Output<float> pred1 = core->predict(samples[1].input).output;

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(pred0[0] > pred1[0], "GPU gdp bright > dark");
}

//===================================================================================================================//

static void testGPUGlobalAvgPoolWithNormAndPool()
{
  std::cout << "--- testGPUGlobalAvgPoolWithNormAndPool (Conv→IN→ReLU→Pool→Conv→ReLU→GAP→Flatten→Dense) ---"
            << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 10, 10};
  config.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig normLayer;
  normLayer.type = CNN::LayerType::INSTANCENORM;
  normLayer.config = CNN::NormLayerConfig{1e-5f, 0.1f};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig poolLayer;
  poolLayer.type = CNN::LayerType::POOL;
  poolLayer.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};

  CNN::CNNLayerConfig conv2;
  conv2.type = CNN::LayerType::CONV;
  conv2.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu2;
  relu2.type = CNN::LayerType::RELU;
  relu2.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig gapLayer;
  gapLayer.type = CNN::LayerType::GLOBALAVGPOOL;
  gapLayer.config = CNN::GlobalAvgPoolLayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {conv1, normLayer, relu1, poolLayer, conv2, relu2, gapLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv1;
  initConv1.numFilters = 4;
  initConv1.inputC = 1;
  initConv1.filterH = 3;
  initConv1.filterW = 3;
  initConv1.filters.assign(4 * 1 * 3 * 3, 0.05f);
  initConv1.biases.assign(4, 0.0f);

  CNN::ConvParameters<float> initConv2;
  initConv2.numFilters = 2;
  initConv2.inputC = 4;
  initConv2.filterH = 3;
  initConv2.filterW = 3;
  initConv2.filters.assign(2 * 4 * 3 * 3, 0.05f);
  initConv2.biases.assign(2, 0.0f);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.shuffleSeed = 42; // Fully deterministic — no retry loop.
  config.progressReports = 0;
  config.numGPUs = 1;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 10, 10});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 10, 10}, 0.0f);
  samples[1].output = {0.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));
  CNN::Output<float> pred0 = core->predict(samples[0].input).output;
  CNN::Output<float> pred1 = core->predict(samples[1].input).output;

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(pred0[0] > pred1[0], "GPU gap+norm+pool bright > dark");
}

//===================================================================================================================//

//===================================================================================================================//

void runGPUGlobalPoolTests()
{
  testGPUWithGlobalAvgPool();
  testGPUWithGlobalDualPool();
  testGPUGlobalAvgPoolWithNormAndPool();
}
