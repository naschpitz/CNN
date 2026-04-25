#include "test_helpers.hpp"

#include "CNN_Normalization.hpp"

#include <cmath>

//===================================================================================================================//

static void testInstanceNormInference()
{
  std::cout << "--- testInstanceNormInference ---" << std::endl;

  CNN::Shape3D shape{2, 2, 2};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

  CNN::NormParameters<double> params;
  params.numChannels = 2;
  params.gamma = {2.0, 0.5};
  params.beta = {1.0, -1.0};
  params.runningMean = {2.5, 6.5};
  params.runningVar = {1.25, 1.25};

  CNN::NormLayerConfig config;
  config.epsilon = 0.0;

  std::vector<CNN::Tensor3D<double>*> batch = {&input};
  CNN::Normalization<double>::propagate(batch, shape, params, config, CNN::LayerType::INSTANCENORM, false);

  CHECK(input.shape.c == 2 && input.shape.h == 2 && input.shape.w == 2, "instancenorm inference shape");

  double invStd0 = 1.0 / std::sqrt(1.25);

  CHECK_NEAR(input.data[0], 2.0 * (1.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [0]");
  CHECK_NEAR(input.data[1], 2.0 * (2.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [1]");
  CHECK_NEAR(input.data[2], 2.0 * (3.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [2]");
  CHECK_NEAR(input.data[3], 2.0 * (4.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [3]");

  CHECK_NEAR(input.data[4], 0.5 * (5.0 - 6.5) * invStd0 - 1.0, 1e-9, "instancenorm infer ch1 [0]");
  CHECK_NEAR(input.data[7], 0.5 * (8.0 - 6.5) * invStd0 - 1.0, 1e-9, "instancenorm infer ch1 [3]");

  CHECK_NEAR(params.runningMean[0], 2.5, 1e-9, "instancenorm infer runningMean unchanged");
  CHECK_NEAR(params.runningVar[0], 1.25, 1e-9, "instancenorm infer runningVar unchanged");
}

//===================================================================================================================//

static void testInstanceNormTraining()
{
  std::cout << "--- testInstanceNormTraining ---" << std::endl;

  CNN::Shape3D shape{2, 2, 2};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

  CNN::NormParameters<double> params;
  params.numChannels = 2;
  params.gamma = {1.0, 1.0};
  params.beta = {0.0, 0.0};
  params.runningMean = {0.0, 0.0};
  params.runningVar = {1.0, 1.0};

  CNN::NormLayerConfig config;
  config.epsilon = 0.0;
  config.momentum = 0.1;

  std::vector<double> statsMean, statsVar;
  std::vector<CNN::Tensor3D<double>> xNorm;

  std::vector<CNN::Tensor3D<double>*> batch = {&input};
  CNN::Normalization<double>::propagate(batch, shape, params, config, CNN::LayerType::INSTANCENORM, true, &xNorm,
                                        &statsMean, &statsVar);

  CHECK(input.shape.c == 2 && input.shape.h == 2 && input.shape.w == 2, "instancenorm train shape");

  CHECK_NEAR(statsMean[0], 2.5, 1e-9, "instancenorm train mean ch0");
  CHECK_NEAR(statsVar[0], 1.25, 1e-9, "instancenorm train var ch0");
  CHECK_NEAR(statsMean[1], 6.5, 1e-9, "instancenorm train mean ch1");
  CHECK_NEAR(statsVar[1], 1.25, 1e-9, "instancenorm train var ch1");

  double invStd = 1.0 / std::sqrt(1.25);
  CHECK_NEAR(input.data[0], (1.0 - 2.5) * invStd, 1e-9, "instancenorm train out ch0 [0]");
  CHECK_NEAR(input.data[3], (4.0 - 2.5) * invStd, 1e-9, "instancenorm train out ch0 [3]");

  CHECK_NEAR(xNorm[0].data[0], input.data[0], 1e-9, "instancenorm train xNorm matches out");
  CHECK_NEAR(xNorm[0].data[7], input.data[7], 1e-9, "instancenorm train xNorm matches out ch1");

  CHECK_NEAR(params.runningMean[0], 0.25, 1e-7, "instancenorm train runningMean updated ch0");
  CHECK_NEAR(params.runningMean[1], 0.65, 1e-7, "instancenorm train runningMean updated ch1");
  CHECK_NEAR(params.runningVar[0], 1.025, 1e-7, "instancenorm train runningVar updated ch0");
}

//===================================================================================================================//

static void testInstanceNormBackpropagate()
{
  std::cout << "--- testInstanceNormBackpropagate ---" << std::endl;

  CNN::Shape3D shape{1, 1, 4};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0};

  CNN::NormParameters<double> params;
  params.numChannels = 1;
  params.gamma = {2.0};
  params.beta = {0.5};
  params.runningMean = {0.0};
  params.runningVar = {1.0};

  CNN::NormLayerConfig config;
  config.epsilon = 0.0;
  config.momentum = 0.1;

  std::vector<double> statsMean, statsVar;
  std::vector<CNN::Tensor3D<double>> xNorm;
  std::vector<CNN::Tensor3D<double>*> batch = {&input};
  CNN::Normalization<double>::propagate(batch, shape, params, config, CNN::LayerType::INSTANCENORM, true, &xNorm,
                                        &statsMean, &statsVar);

  CHECK_NEAR(statsMean[0], 2.5, 1e-9, "backprop setup mean");
  CHECK_NEAR(statsVar[0], 1.25, 1e-9, "backprop setup var");

  CNN::Tensor3D<double> dOutput(shape, 1.0);
  std::vector<double> dGamma, dBeta;
  std::vector<CNN::Tensor3D<double>*> dBatch = {&dOutput};
  CNN::Normalization<double>::backpropagate(dBatch, shape, params, config, CNN::LayerType::INSTANCENORM, statsMean,
                                            statsVar, xNorm, dGamma, dBeta);

  CHECK_NEAR(dBeta[0], 4.0, 1e-9, "instancenorm backprop dBeta");
  CHECK_NEAR(dGamma[0], 0.0, 1e-9, "instancenorm backprop dGamma");
  CHECK(dOutput.shape.c == 1 && dOutput.shape.h == 1 && dOutput.shape.w == 4, "instancenorm backprop shape");

  for (ulong i = 0; i < 4; i++)
    CHECK_NEAR(dOutput.data[i], 0.0, 1e-9, "instancenorm backprop dInput uniform=0");
}

//===================================================================================================================//

static void testInstanceNormBackpropGradient()
{
  std::cout << "--- testInstanceNormBackpropGradient ---" << std::endl;

  CNN::Shape3D shape{1, 1, 4};
  CNN::Tensor3D<double> inputOrig(shape);
  inputOrig.data = {1.0, 2.0, 3.0, 4.0};

  CNN::NormParameters<double> params;
  params.numChannels = 1;
  params.gamma = {1.0};
  params.beta = {0.0};
  params.runningMean = {0.0};
  params.runningVar = {1.0};

  CNN::NormLayerConfig config;
  config.epsilon = 1e-7;
  config.momentum = 0.1;

  CNN::Tensor3D<double> input(shape);
  input.data = inputOrig.data;
  std::vector<double> statsMean, statsVar;
  std::vector<CNN::Tensor3D<double>> xNorm;
  std::vector<CNN::Tensor3D<double>*> batch = {&input};
  CNN::Normalization<double>::propagate(batch, shape, params, config, CNN::LayerType::INSTANCENORM, true, &xNorm,
                                        &statsMean, &statsVar);

  CNN::Tensor3D<double> dOutput(shape);
  dOutput.data = {0.1, 0.2, 0.3, 0.4};

  std::vector<double> dGamma, dBeta;
  std::vector<CNN::Tensor3D<double>*> dBatch = {&dOutput};
  CNN::Normalization<double>::backpropagate(dBatch, shape, params, config, CNN::LayerType::INSTANCENORM, statsMean,
                                            statsVar, xNorm, dGamma, dBeta);

  CHECK_NEAR(dBeta[0], 1.0, 1e-9, "grad dBeta");

  double eps = 1e-5;

  for (ulong i = 0; i < 4; i++) {
    CNN::Tensor3D<double> inputPlus(shape);
    inputPlus.data = inputOrig.data;
    inputPlus.data[i] += eps;

    CNN::NormParameters<double> pPlus = params;
    pPlus.runningMean = {0.0};
    pPlus.runningVar = {1.0};
    std::vector<double> smP, svP;
    std::vector<CNN::Tensor3D<double>> xnP;
    std::vector<CNN::Tensor3D<double>*> bP = {&inputPlus};
    CNN::Normalization<double>::propagate(bP, shape, pPlus, config, CNN::LayerType::INSTANCENORM, true, &xnP, &smP,
                                          &svP);

    CNN::Tensor3D<double> inputMinus(shape);
    inputMinus.data = inputOrig.data;
    inputMinus.data[i] -= eps;

    CNN::NormParameters<double> pMinus = params;
    pMinus.runningMean = {0.0};
    pMinus.runningVar = {1.0};
    std::vector<double> smM, svM;
    std::vector<CNN::Tensor3D<double>> xnM;
    std::vector<CNN::Tensor3D<double>*> bM = {&inputMinus};
    CNN::Normalization<double>::propagate(bM, shape, pMinus, config, CNN::LayerType::INSTANCENORM, true, &xnM, &smM,
                                          &svM);

    double numGrad = 0.0;

    for (ulong j = 0; j < 4; j++)
      numGrad += 0.1 * (j + 1) * (inputPlus.data[j] - inputMinus.data[j]) / (2.0 * eps);

    CHECK_NEAR(dOutput.data[i], numGrad, 1e-4, "instancenorm numerical gradient check");
  }
}

//===================================================================================================================//

void runInstanceNormTests()
{
  testInstanceNormInference();
  testInstanceNormTraining();
  testInstanceNormBackpropagate();
  testInstanceNormBackpropGradient();
}