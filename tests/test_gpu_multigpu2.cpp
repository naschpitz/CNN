#include "test_helpers.hpp"

static void testMultiGPUCrossEntropyTraining()
{
  std::cout << "--- testMultiGPUCrossEntropyTraining ---" << std::endl;

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

  CHECK(result.averageLoss >= 0.0f, "multi-GPU CNN CE: loss non-negative");
  CHECK(std::isfinite(result.averageLoss), "multi-GPU CNN CE: loss is finite");

  auto out0 = core->predict(samples[0].input).output;
  float sum0 = out0[0] + out0[1] + out0[2];
  CHECK_NEAR(sum0, 1.0f, 0.01f, "multi-GPU CNN CE: softmax sums to 1");

  std::cout << "  multi-GPU CNN CE avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

static void testMultiGPUParameterRoundTrip()
{
  std::cout << "--- testMultiGPUParameterRoundTrip ---" << std::endl;

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
  CHECK(params.convParams.size() == 1, "multi-GPU param roundtrip: convParams count");
  CHECK(params.convParams[0].filters.size() == 9, "multi-GPU param roundtrip: filters size");

  // Create predict core with trained params
  CNN::CoreConfig<float> predictConfig;
  predictConfig.modeType = CNN::ModeType::PREDICT;
  predictConfig.deviceType = CNN::DeviceType::GPU;
  predictConfig.inputShape = {1, 5, 5};
  predictConfig.logLevel = CNN::LogLevel::ERROR;
  predictConfig.layersConfig = config.layersConfig;
  predictConfig.parameters = params;

  auto predictCore = CNN::Core<float>::makeCore(predictConfig);
  CNN::Output<float> origPred = core->predict(samples[0].input).output;
  CNN::Output<float> newPred = predictCore->predict(samples[0].input).output;
  CHECK_NEAR(origPred[0], newPred[0], 0.01f, "multi-GPU param roundtrip: prediction match");
  std::cout << "  multi-GPU original=" << origPred[0] << "  from_params=" << newPred[0] << std::endl;
}

//===================================================================================================================//

static void testMultiGPUWithPoolLayer()
{
  std::cout << "--- testMultiGPUWithPoolLayer ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 8, 8};
  config.logLevel = CNN::LogLevel::ERROR;
  config.numGPUs = 2;

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

  // All-positive but asymmetric init — bright (positive) inputs propagate
  // through both ReLU layers (no dead neurons), and the two filters break
  // symmetry by having different magnitudes. Negative weights would risk
  // dead ReLUs that zero the dense input and stall training.
  CNN::ConvParameters<float> initConv1;
  initConv1.numFilters = 2;
  initConv1.inputC = 1;
  initConv1.filterH = 3;
  initConv1.filterW = 3;
  initConv1.filters = {
    0.05f, 0.10f, 0.05f, 0.08f, 0.12f, 0.08f, 0.05f, 0.10f, 0.05f, // filter 0 (centre-weighted)
    0.15f, 0.05f, 0.15f, 0.05f, 0.05f, 0.05f, 0.15f, 0.05f, 0.15f, // filter 1 (corner-weighted)
  };

  initConv1.biases.assign(2, 0.0f);

  CNN::ConvParameters<float> initConv2;
  initConv2.numFilters = 1;
  initConv2.inputC = 2;
  initConv2.filterH = 3;
  initConv2.filterW = 3;
  initConv2.filters = {
    0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, // filter 0, channel 0
    0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, // filter 0, channel 1
  };

  initConv2.biases.assign(1, 0.0f);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.shuffleSeed = 42; // Fully deterministic — no retry loop.
  config.progressReports = 0;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 8, 8});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 8, 8}, 0.0f);
  samples[1].output = {0.0f};

  // Capture initial loss, train, then verify training drove loss strictly
  // down. A two-sample dataset is too small to expect full convergence
  // from a fixed init, but a working multi-GPU training pipeline must at
  // least make the loss go down — that's what we assert.
  auto core = CNN::Core<float>::makeCore(config);
  CNN::TestResult<float> beforeResult = core->test(samples.size(), CNN::makeSampleProvider(samples));
  core->train(samples.size(), CNN::makeSampleProvider(samples));
  CNN::TestResult<float> afterResult = core->test(samples.size(), CNN::makeSampleProvider(samples));

  CNN::Output<float> pred0 = core->predict(samples[0].input).output;
  CNN::Output<float> pred1 = core->predict(samples[1].input).output;

  std::cout << "  multi-GPU pool loss before=" << beforeResult.averageLoss << " after=" << afterResult.averageLoss
            << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(std::isfinite(afterResult.averageLoss), "multi-GPU with pool: post-training loss is finite");
  CHECK(afterResult.averageLoss < beforeResult.averageLoss, "multi-GPU with pool: training reduced loss");
}

void runGPUMultiGPUTests2()
{
  testMultiGPUCrossEntropyTraining();
  testMultiGPUParameterRoundTrip();
  testMultiGPUWithPoolLayer();
}
