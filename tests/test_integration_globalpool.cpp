#include "test_helpers.hpp"

//===================================================================================================================//

static void testGlobalAvgPoolEndToEnd()
{
  std::cout << "--- testGlobalAvgPoolEndToEnd ---" << std::endl;

  // 1x5x5 → Conv(2 filters 3x3 valid) → 2x3x3 → ReLU → GlobalAvgPool → 2x1x1 → Flatten(2) → Dense(1, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig gapLayer;
  gapLayer.type = CNN::LayerType::GLOBALAVGPOOL;
  gapLayer.config = CNN::GlobalAvgPoolLayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, gapLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 2;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(2 * 1 * 3 * 3, 0.1);
  initConv.biases.assign(2, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

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

  CHECK(core != nullptr, "gap core creation");
  CHECK(pred0.size() == 1, "gap predict output size 0");
  CHECK(pred1.size() == 1, "gap predict output size 1");
  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "gap: bright > dark after training (5 attempts)");
}

//===================================================================================================================//

static void testGlobalAvgPoolWithNorm()
{
  std::cout << "--- testGlobalAvgPoolWithNorm ---" << std::endl;

  // Conv → InstanceNorm → ReLU → GlobalAvgPool → Flatten → Dense
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 6, 6};
  config.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig normLayer;
  normLayer.type = CNN::LayerType::INSTANCENORM;
  normLayer.config = CNN::NormLayerConfig{1e-5f, 0.1f};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig gapLayer;
  gapLayer.type = CNN::LayerType::GLOBALAVGPOOL;
  gapLayer.config = CNN::GlobalAvgPoolLayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, normLayer, reluLayer, gapLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 2;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(2 * 1 * 3 * 3, 0.1);
  initConv.biases.assign(2, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 150;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 6, 6});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 6, 6}, 0.0);
  samples[1].output = {0.0};

  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    auto pred0 = core->predict(samples[0].input);
    auto pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  CHECK(converged, "gap+norm: bright > dark after training (5 attempts)");
}

//===================================================================================================================//

static void testGlobalAvgPoolAfterPool()
{
  std::cout << "--- testGlobalAvgPoolAfterPool ---" << std::endl;

  // Conv → ReLU → Pool(2x2) → GlobalAvgPool → Flatten → Dense
  // This tests GAP receiving already-reduced spatial dims
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 8, 8};
  config.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig poolLayer;
  poolLayer.type = CNN::LayerType::POOL;
  poolLayer.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};

  CNN::CNNLayerConfig gapLayer;
  gapLayer.type = CNN::LayerType::GLOBALAVGPOOL;
  gapLayer.config = CNN::GlobalAvgPoolLayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, poolLayer, gapLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 4;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(4 * 1 * 3 * 3, 0.05);
  initConv.biases.assign(4, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 8, 8});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 8, 8}, 0.0);
  samples[1].output = {0.0};

  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    auto pred0 = core->predict(samples[0].input);
    auto pred1 = core->predict(samples[1].input);

    if (pred0[0] > pred1[0])
      converged = true;
  }

  CHECK(converged, "gap+pool: bright > dark after training (5 attempts)");
}

//===================================================================================================================//

//===================================================================================================================//

static void testBatchPredict()
{
  std::cout << "--- testBatchPredict ---" << std::endl;

  // 1x5x5 → Conv(1 filter 3x3 valid) → ReLU → Flatten(9) → Dense(1, sigmoid)
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
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  // "bright" → 1, "dark" → 0
  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  std::unique_ptr<CNN::Core<double>> core;
  bool converged = false;
  CNN::Outputs<double> batchOutputs;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;

    core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));

    // Batch predict both inputs at once
    CNN::Inputs<double> inputs = {samples[0].input, samples[1].input};
    batchOutputs = core->predict(inputs);

    if (batchOutputs[0][0] > batchOutputs[1][0])
      converged = true;
  }

  CHECK(batchOutputs.size() == 2, "batch predict returns 2 outputs");
  CHECK(batchOutputs[0].size() == 1, "output[0] size = 1");
  CHECK(batchOutputs[1].size() == 1, "output[1] size = 1");
  CHECK(converged, "batch predict: bright > dark after training");

  // Verify batch predict matches single predict
  CNN::Output<double> single0 = core->predict(samples[0].input);
  CNN::Output<double> single1 = core->predict(samples[1].input);
  CHECK_NEAR(batchOutputs[0][0], single0[0], 1e-10, "batch[0] matches single predict");
  CHECK_NEAR(batchOutputs[1][0], single1[0], 1e-10, "batch[1] matches single predict");
}

//===================================================================================================================//

static void testGlobalDualPoolEndToEnd()
{
  std::cout << "--- testGlobalDualPoolEndToEnd ---" << std::endl;

  // 1x5x5 → Conv(2 filters 3x3 valid) → 2x3x3 → ReLU → GlobalDualPool → 4x1x1 → Flatten(4) → Dense(1, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
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

  config.layersConfig.cnnLayers = {conv1, relu1, gdpLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<double> initConv1;
  initConv1.numFilters = 2;
  initConv1.inputC = 1;
  initConv1.filterH = 3;
  initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1);
  initConv1.biases.assign(2, 0.0);

  config.parameters.convParams = {initConv1};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

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
  CHECK(converged, "GDP end-to-end: bright > dark (5 attempts)");
}

//===================================================================================================================//


//===================================================================================================================//

void runIntegrationGlobalPoolTests()
{
  testGlobalAvgPoolEndToEnd();
  testGlobalAvgPoolWithNorm();
  testGlobalAvgPoolAfterPool();
  testGlobalDualPoolEndToEnd();
}
