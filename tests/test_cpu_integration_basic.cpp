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
    pred0 = core->predict(samples[0].input).output;
    pred1 = core->predict(samples[1].input).output;

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
    pred0 = core->predict(samples[0].input).output;
    pred1 = core->predict(samples[1].input).output;

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
    pred0 = core->predict(samples[0].input).output;
    pred1 = core->predict(samples[1].input).output;

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
    pred0 = core->predict(samples[0].input).output;
    pred1 = core->predict(samples[1].input).output;

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "multichannel bright > dark (5 attempts)");
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
  CNN::PredictResults<double> batchResults;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;

    core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));

    // Batch predict both inputs at once
    CNN::Inputs<double> inputs = {samples[0].input, samples[1].input};
    batchResults = core->predict(inputs);

    if (batchResults[0].output[0] > batchResults[1].output[0])
      converged = true;
  }

  CHECK(batchResults.size() == 2, "batch predict returns 2 outputs");
  CHECK(batchResults[0].output.size() == 1, "output[0] size = 1");
  CHECK(batchResults[1].output.size() == 1, "output[1] size = 1");
  CHECK(batchResults[0].logits.size() == batchResults[0].output.size(), "logits[0] size matches output");
  CHECK(batchResults[1].logits.size() == batchResults[1].output.size(), "logits[1] size matches output");
  CHECK(converged, "batch predict: bright > dark after training");

  // Verify batch predict matches single predict
  CNN::Output<double> single0 = core->predict(samples[0].input).output;
  CNN::Output<double> single1 = core->predict(samples[1].input).output;
  CHECK_NEAR(batchResults[0].output[0], single0[0], 1e-10, "batch[0] matches single predict");
  CHECK_NEAR(batchResults[1].output[0], single1[0], 1e-10, "batch[1] matches single predict");
}

//===================================================================================================================//

//===================================================================================================================//

void runIntegrationBasicTests()
{
  testEndToEnd();
  testMultiConvStack();
  testConvPoolConv();
  testMultiChannelInput();
  testBatchPredict();
}
