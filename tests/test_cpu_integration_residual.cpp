#include "test_helpers.hpp"

//===================================================================================================================//

static void testResidualIdentityEndToEnd()
{
  std::cout << "--- testResidualIdentityEndToEnd ---" << std::endl;

  // residual_start → Conv(4, same) → ReLU → Conv(4, same) → ReLU → residual_end → GAP → Flatten → Dense(1)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig resStart;
  resStart.type = CNN::LayerType::RESIDUAL_START;
  resStart.config = CNN::ResidualStartConfig{};

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

  CNN::CNNLayerConfig relu;
  relu.type = CNN::LayerType::RELU;
  relu.config = CNN::ReLULayerConfig{};

  // Second conv must also output 4 channels (same as input from res_start after first conv)
  // Wait — the input to residual_start is 1ch, but block output is 4ch. That's a projection case!
  // For identity, we need the first conv BEFORE residual_start.
  // Let's restructure: Conv(4) → residual_start → Conv(4, same) → ReLU → residual_end → GAP → Dense
  CNN::CNNLayerConfig conv2;
  conv2.type = CNN::LayerType::CONV;
  conv2.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

  CNN::CNNLayerConfig resEnd;
  resEnd.type = CNN::LayerType::RESIDUAL_END;
  resEnd.config = CNN::ResidualEndConfig{};

  CNN::CNNLayerConfig gapLayer;
  gapLayer.type = CNN::LayerType::GLOBALAVGPOOL;
  gapLayer.config = CNN::GlobalAvgPoolLayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {conv1, relu, resStart, conv2, relu, resEnd, gapLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  bool converged = false;
  CNN::Output<double> pred0, pred1;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input).output;
    pred1 = core->predict(samples[1].input).output;

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "Residual identity end-to-end: bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testResidualParametersEndToEnd()
{
  std::cout << "--- testResidualParametersEndToEnd ---" << std::endl;

  // residual_start (1ch) → Conv(4, valid) → ReLU → residual_end (4ch, projection from 1→4)
  // → GAP → Flatten → Dense(1)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 8, 8};
  config.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig resStart;
  resStart.type = CNN::LayerType::RESIDUAL_START;
  resStart.config = CNN::ResidualStartConfig{};

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

  CNN::CNNLayerConfig relu;
  relu.type = CNN::LayerType::RELU;
  relu.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig resEnd;
  resEnd.type = CNN::LayerType::RESIDUAL_END;
  resEnd.config = CNN::ResidualEndConfig{};

  CNN::CNNLayerConfig gapLayer;
  gapLayer.type = CNN::LayerType::GLOBALAVGPOOL;
  gapLayer.config = CNN::GlobalAvgPoolLayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {resStart, conv1, relu, resEnd, gapLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 8, 8});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 8, 8}, 0.0);
  samples[1].output = {0.0};

  bool converged = false;
  CNN::Output<double> pred0, pred1;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input).output;
    pred1 = core->predict(samples[1].input).output;

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "Residual projection end-to-end: bright > dark (5 attempts)");
}

//===================================================================================================================//

static void testResidualMixedIdentityProjectionEndToEnd()
{
  std::cout << "--- testResidualMixedIdentityProjectionEndToEnd ---" << std::endl;

  // stem(4,stride2) → res_identity(4→4) → pool → res_projection(4→8) → GAP → Flatten → Dense(1)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 16, 16};
  config.logLevel = CNN::LogLevel::ERROR;

  CNN::CNNLayerConfig stem;
  stem.type = CNN::LayerType::CONV;
  stem.config = CNN::ConvLayerConfig{4, 3, 3, 2, 2, CNN::SlidingStrategyType::SAME};

  CNN::CNNLayerConfig relu;
  relu.type = CNN::LayerType::RELU;
  relu.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig resStart;
  resStart.type = CNN::LayerType::RESIDUAL_START;
  resStart.config = CNN::ResidualStartConfig{};

  CNN::CNNLayerConfig conv4;
  conv4.type = CNN::LayerType::CONV;
  conv4.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

  CNN::CNNLayerConfig resEnd;
  resEnd.type = CNN::LayerType::RESIDUAL_END;
  resEnd.config = CNN::ResidualEndConfig{};

  CNN::CNNLayerConfig pool;
  pool.type = CNN::LayerType::POOL;
  pool.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};

  CNN::CNNLayerConfig conv8;
  conv8.type = CNN::LayerType::CONV;
  conv8.config = CNN::ConvLayerConfig{8, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

  CNN::CNNLayerConfig gapLayer;
  gapLayer.type = CNN::LayerType::GLOBALAVGPOOL;
  gapLayer.config = CNN::GlobalAvgPoolLayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  // stem → identity res block → pool → projection res block → GAP → flatten
  config.layersConfig.cnnLayers = {stem,     relu,  resStart, conv4,  relu,     resEnd,      pool,
                                   resStart, conv8, relu,     resEnd, gapLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 16, 16});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 16, 16}, 0.0);
  samples[1].output = {0.0};

  bool converged = false;
  CNN::Output<double> pred0, pred1;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples.size(), CNN::makeSampleProvider(samples));
    pred0 = core->predict(samples[0].input).output;
    pred1 = core->predict(samples[1].input).output;

    if (pred0[0] > pred1[0])
      converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "Mixed identity+projection residual e2e: bright > dark (5 attempts)");
}

//===================================================================================================================//

void runIntegrationResidualTests()
{
  testResidualIdentityEndToEnd();
  testResidualParametersEndToEnd();
  testResidualMixedIdentityProjectionEndToEnd();
}
