#include "test_helpers.hpp"

static void testParameterRoundTrip()
{
  std::cout << "--- testParameterRoundTrip ---" << std::endl;

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

  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  auto core = CNN::Core<double>::makeCore(config);

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  core->train(samples.size(), CNN::makeSampleProvider(samples));

  // Get trained parameters
  const CNN::Parameters<double>& params = core->getParameters();
  CHECK(params.convParams.size() == 1, "param roundtrip convParams count");
  CHECK(params.convParams[0].filters.size() == 9, "param roundtrip filters size");
  CHECK(params.convParams[0].biases.size() == 1, "param roundtrip biases size");

  // Parameters should have changed from initial values after training
  bool filtersChanged = false;

  for (ulong i = 0; i < 9; i++) {
    if (std::fabs(params.convParams[0].filters[i] - 0.1) > 1e-6) {
      filtersChanged = true;
      break;
    }
  }

  CHECK(filtersChanged, "param roundtrip: filters changed after training");

  // Create a new core in PREDICT mode with trained parameters
  CNN::CoreConfig<double> predictConfig;
  predictConfig.modeType = CNN::ModeType::PREDICT;
  predictConfig.deviceType = CNN::DeviceType::CPU;
  predictConfig.inputShape = {1, 5, 5};
  predictConfig.logLevel = CNN::LogLevel::ERROR;
  predictConfig.layersConfig = config.layersConfig;
  predictConfig.parameters = params;

  auto predictCore = CNN::Core<double>::makeCore(predictConfig);
  CHECK(predictCore != nullptr, "param roundtrip predict core creation");

  // Predictions should match
  CNN::Output<double> origPred = core->predict(samples[0].input).output;
  CNN::Output<double> newPred = predictCore->predict(samples[0].input).output;
  CHECK_NEAR(origPred[0], newPred[0], 1e-9, "param roundtrip prediction match");
  std::cout << "  original=" << origPred[0] << "  from_params=" << newPred[0] << std::endl;
}

static void testParametersDuringTraining()
{
  std::cout << "--- testParametersDuringTraining ---" << std::endl;

  // Train and verify getParameters() returns populated conv AND dense params
  // during training (in the callback), not just after training ends.
  // This test would FAIL without the dense params sync fix in CNN_CoreCPU.cpp.
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

  config.trainingConfig.numEpochs = 10;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  auto core = CNN::Core<double>::makeCore(config);

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  bool paramsChecked = false;
  bool convFiltersNonEmpty = false;
  bool denseWeightsNonEmpty = false;
  bool denseBiasesNonEmpty = false;
  ulong lastEpoch = 0;

  core->setTrainingCallback([&](const CNN::TrainingProgress<double>& progress) {
    // Detect epoch transition (first callback of a new epoch)
    if (progress.currentEpoch > lastEpoch && lastEpoch > 0 && !paramsChecked) {
      const CNN::Parameters<double>& params = core->getParameters();
      convFiltersNonEmpty = !params.convParams.empty() && !params.convParams[0].filters.empty();
      denseWeightsNonEmpty = !params.denseParams.weights.empty();
      denseBiasesNonEmpty = !params.denseParams.biases.empty();
      paramsChecked = true;
    }

    lastEpoch = progress.currentEpoch;
  });

  core->train(samples.size(), CNN::makeSampleProvider(samples));

  CHECK(paramsChecked, "epoch transition detected in callback");
  CHECK(convFiltersNonEmpty, "convParams[0].filters non-empty during training");
  CHECK(denseWeightsNonEmpty, "denseParams.weights non-empty during training");
  CHECK(denseBiasesNonEmpty, "denseParams.biases non-empty during training");
}

//===================================================================================================================//

static void testMultipleOutputNeurons()
{
  std::cout << "--- testMultipleOutputNeurons ---" << std::endl;

  // 1x8x8 → Conv(2,3x3) → ReLU → Flatten(72) → Dense(3, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 8, 8};
  config.logLevel = CNN::LogLevel::ERROR;

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

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 2;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(2 * 9, 0.1);
  initConv.biases.assign(2, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.shuffleSeed = 42; // Fully deterministic — no retry loop.
  config.numThreads = 1; // Single-threaded — parallel batch reduction order is FP-non-deterministic.
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 8, 8});
  samples[0].output = {1.0, 0.0, 1.0}; // target: [1, 0, 1]
  samples[1].input = CNN::Tensor3D<double>({1, 8, 8}, 0.0);
  samples[1].output = {0.0, 1.0, 0.0}; // target: [0, 1, 0]

  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));
  CNN::Output<double> pred0 = core->predict(samples[0].input).output;
  CNN::Output<double> pred1 = core->predict(samples[1].input).output;

  std::cout << "  pred(bright)=[" << pred0[0] << "," << pred0[1] << "," << pred0[2] << "]" << std::endl;
  CHECK(pred0.size() == 3, "multi-output size");
  CHECK(pred0[0] > pred1[0], "multi-output[0] bright > dark");
}

//===================================================================================================================//

void runIntegrationBasicTests2()
{
  testParameterRoundTrip();
  testParametersDuringTraining();
  testMultipleOutputNeurons();
}
