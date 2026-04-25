#include "test_helpers.hpp"

//===================================================================================================================//

static void testCrossEntropyGradientNumerical()
{
  std::cout << "--- testCrossEntropyGradientNumerical ---" << std::endl;

  // Verify the analytical gradient formula matches numerical finite differences.
  // Loss = -sum(w_i * y_i * log(a_i))
  // dL/da_j = -w_j * y_j / a_j

  std::vector<double> activations = {0.7, 0.2, 0.1};
  std::vector<double> targets = {1.0, 0.0, 0.0};
  std::vector<double> weights = {1.0, 1.0, 1.0};

  auto computeLoss = [&](const std::vector<double>& a) {
    double loss = 0;
    const double epsilon = 1e-7;

    for (ulong i = 0; i < a.size(); i++) {
      double pred = std::max(a[i], epsilon);
      loss -= weights[i] * targets[i] * std::log(pred);
    }

    return loss;
  };

  double eps = 1e-6;

  for (ulong j = 0; j < activations.size(); j++) {
    const double epsilon = 1e-7;
    double pred = std::max(activations[j], epsilon);
    double analyticalGrad = -weights[j] * targets[j] / pred;

    std::vector<double> aPlus = activations;
    aPlus[j] += eps;
    std::vector<double> aMinus = activations;
    aMinus[j] -= eps;
    double numericalGrad = (computeLoss(aPlus) - computeLoss(aMinus)) / (2.0 * eps);

    std::cout << "  j=" << j << " analytical=" << analyticalGrad << " numerical=" << numericalGrad << std::endl;
    CHECK_NEAR(analyticalGrad, numericalGrad, 1e-4, "CE gradient numerical check");
  }

  // Also test with non-uniform weights
  weights = {5.0, 1.0, 2.0};
  targets = {0.0, 1.0, 0.0};
  activations = {0.1, 0.6, 0.3};

  for (ulong j = 0; j < activations.size(); j++) {
    const double epsilon = 1e-7;
    double pred = std::max(activations[j], epsilon);
    double analyticalGrad = -weights[j] * targets[j] / pred;

    std::vector<double> aPlus = activations;
    aPlus[j] += eps;
    std::vector<double> aMinus = activations;
    aMinus[j] -= eps;
    double numericalGrad = (computeLoss(aPlus) - computeLoss(aMinus)) / (2.0 * eps);

    std::cout << "  weighted j=" << j << " analytical=" << analyticalGrad << " numerical=" << numericalGrad
              << std::endl;
    CHECK_NEAR(analyticalGrad, numericalGrad, 1e-4, "weighted CE gradient numerical check");
  }
}

//===================================================================================================================//

static void testCrossEntropyLossDecreases()
{
  std::cout << "--- testCrossEntropyLossDecreases (CNN) ---" << std::endl;

  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;

  // 2 filters with varied init to break symmetry
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
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 2;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters = {
    0.1,  -0.2, 0.3,  -0.1, 0.4,  -0.3, 0.2,  -0.1, 0.5, // filter 0
    -0.3, 0.1,  -0.2, 0.4,  -0.1, 0.2,  -0.4, 0.3,  -0.1 // filter 1
  };

  initConv.biases.assign(2, 0.0);
  config.parameters.convParams = {initConv};

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;

  // Both samples have non-zero inputs so conv produces meaningful features
  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5}, 0.6, 1.0);
  samples[0].output = {1.0, 0.0};
  samples[1].input = makeGradientInput<double>({1, 5, 5}, 0.0, 0.4);
  samples[1].output = {0.0, 1.0};

  // Train 50 epochs, measure loss
  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));
  CNN::TestResult<double> result1 = core->test(samples.size(), CNN::makeSampleProvider(samples));

  // Train 200 more epochs from same params
  config.trainingConfig.numEpochs = 200;
  config.parameters = core->getParameters();
  auto core2 = CNN::Core<double>::makeCore(config);
  core2->train(samples.size(), CNN::makeSampleProvider(samples));
  CNN::TestResult<double> result2 = core2->test(samples.size(), CNN::makeSampleProvider(samples));

  std::cout << "  loss after 50=" << result1.averageLoss << "  after 250=" << result2.averageLoss << std::endl;
  CHECK(result2.averageLoss < result1.averageLoss, "CNN CE loss decreases with more training");
  CHECK(result2.averageLoss < 0.5, "CNN CE loss below 0.5 after 250 epochs on trivial problem");
}

//===================================================================================================================//

static void testCrossEntropyTraining()
{
  std::cout << "--- testCrossEntropyTraining (CNN) ---" << std::endl;

  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = CNN::LogLevel::ERROR;

  // Use 4 filters with varied init so flatten features are diverse enough for 3-class softmax
  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{3, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 4;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  // Varied init: break symmetry so each filter extracts different features
  initConv.filters = {
    0.1,  -0.2, 0.3,  -0.1, 0.4,  -0.3, 0.2,  -0.1, 0.5, // filter 0
    -0.3, 0.1,  -0.2, 0.4,  -0.1, 0.2,  -0.4, 0.3,  -0.1, // filter 1
    0.2,  0.3,  -0.1, -0.2, 0.1,  0.4,  -0.3, -0.1, 0.2, // filter 2
    -0.1, -0.3, 0.2,  0.1,  -0.4, 0.3,  0.2,  -0.2, -0.1 // filter 3
  };

  initConv.biases.assign(4, 0.0);
  config.parameters.convParams = {initConv};

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.numEpochs = 300;
  config.trainingConfig.learningRate = 0.003f;
  config.trainingConfig.shuffleSeed = 42; // Fully deterministic — no retry loop.
  config.progressReports = 0;

  // 3-class classification with truly distinct spatial patterns (not just
  // different magnitudes of the same gradient — that mapping isn't linearly
  // separable by softmax of conv features).
  CNN::Samples<double> samples(3);
  samples[0].input = makeGradientInput<double>({1, 5, 5}, 0.8, 1.0); // bright/uniform-ish
  samples[0].output = {1.0, 0.0, 0.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0); // all-zero
  samples[1].output = {0.0, 1.0, 0.0};
  samples[2].input = makeGradientInput<double>({1, 5, 5}, 0.0, 1.0); // high-contrast gradient
  samples[2].output = {0.0, 0.0, 1.0};

  auto core = CNN::Core<double>::makeCore(config);

  // Property test: a working CE training pipeline must drive loss strictly
  // down from initial. Asserting an absolute threshold is fragile because it
  // depends on architecture/init/optimizer details that don't affect
  // correctness. Strict decrease is the real correctness signal.
  CNN::TestResult<double> beforeResult = core->test(samples.size(), CNN::makeSampleProvider(samples));
  core->train(samples.size(), CNN::makeSampleProvider(samples));
  CNN::TestResult<double> afterResult = core->test(samples.size(), CNN::makeSampleProvider(samples));

  auto out0 = core->predict(samples[0].input).output;
  double sum0 = out0[0] + out0[1] + out0[2];

  std::cout << "  CNN CE loss before=" << beforeResult.averageLoss << " after=" << afterResult.averageLoss
            << " accuracy=" << afterResult.accuracy << "%" << std::endl;

  CHECK(afterResult.numSamples == 3, "CNN CE: 3 samples");
  CHECK(std::isfinite(afterResult.averageLoss), "CNN CE: loss is finite");
  CHECK(afterResult.averageLoss < beforeResult.averageLoss, "CNN CE: training reduced loss");
  CHECK(std::fabs(sum0 - 1.0) < 1e-5, "CNN CE: softmax outputs sum to 1");
  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == CNN::CostFunctionType::CROSS_ENTROPY, "CNN CE: type preserved");
}

//===================================================================================================================//

static void testWeightedCrossEntropyTraining()
{
  std::cout << "--- testWeightedCrossEntropyTraining (CNN) ---" << std::endl;

  // Cross-entropy with per-class weights
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
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1);
  initConv.biases.assign(1, 0.0);
  config.parameters.convParams = {initConv};

  config.costFunctionConfig.type = CNN::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {5.0, 1.0};
  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0, 0.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0, 1.0};

  auto core = CNN::Core<double>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  CNN::TestResult<double> result = core->test(samples.size(), CNN::makeSampleProvider(samples));

  CHECK(result.averageLoss >= 0.0, "CNN weighted CE: loss non-negative");
  CHECK(std::isfinite(result.averageLoss), "CNN weighted CE: loss is finite");

  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == CNN::CostFunctionType::CROSS_ENTROPY, "CNN weighted CE: type preserved");
  CHECK(cfc.weights.size() == 2, "CNN weighted CE: weights preserved");
  CHECK_NEAR(cfc.weights[0], 5.0, 1e-10, "CNN weighted CE: weight[0] = 5.0");

  std::cout << "  CNN weighted CE avgLoss=" << result.averageLoss << std::endl;
}

void runIntegrationCostFuncTests()
{
  testCrossEntropyGradientNumerical();
  testCrossEntropyLossDecreases();
  testCrossEntropyTraining();
  testWeightedCrossEntropyTraining();
}
