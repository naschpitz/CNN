#include "test_helpers.hpp"
#include "CNN_Device.hpp"
#include "CNN_Mode.hpp"
#include "CNN_PoolType.hpp"
#include "CNN_CostFunctionConfig.hpp"
#include "CNN_GlobalAvgPool.hpp"
#include "CNN_GlobalDualPool.hpp"
#include "CNN_Normalization.hpp"

//===================================================================================================================//

static void testTensor3D()
{
  std::cout << "--- testTensor3D ---" << std::endl;

  CNN::Tensor3D<double> t({2, 3, 4});
  CHECK(t.shape.c == 2 && t.shape.h == 3 && t.shape.w == 4, "shape");
  CHECK(t.size() == 24, "size");
  CHECK(t.at(0, 0, 0) == 0.0, "zero-initialized");

  t.at(1, 2, 3) = 42.0;
  CHECK_NEAR(t.at(1, 2, 3), 42.0, 1e-9, "at accessor");

  CNN::Tensor3D<double> t2({1, 2, 2}, 5.0);
  CHECK_NEAR(t2.at(0, 1, 1), 5.0, 1e-9, "fill constructor");
}

//===================================================================================================================//

static void testReLU()
{
  std::cout << "--- testReLU ---" << std::endl;

  CNN::Tensor3D<double> input({1, 2, 3});
  input.data = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

  CNN::Tensor3D<double> out = CNN::ReLU<double>::propagate(input);
  CHECK_NEAR(out.data[0], 0.0, 1e-9, "relu neg -> 0");
  CHECK_NEAR(out.data[3], 1.0, 1e-9, "relu pos passthrough");
  CHECK_NEAR(out.data[5], 3.0, 1e-9, "relu pos passthrough");

  CNN::Tensor3D<double> dOut({1, 2, 3}, 1.0);
  CNN::Tensor3D<double> dIn = CNN::ReLU<double>::backpropagate(dOut, input);
  CHECK_NEAR(dIn.data[0], 0.0, 1e-9, "relu backprop blocks neg");
  CHECK_NEAR(dIn.data[3], 1.0, 1e-9, "relu backprop passes pos");
}

//===================================================================================================================//

static void testMaxPool()
{
  std::cout << "--- testMaxPool ---" << std::endl;

  CNN::Tensor3D<double> input({1, 4, 4});

  for (ulong i = 0; i < 16; i++)
    input.data[i] = static_cast<double>(i + 1);

  CNN::PoolLayerConfig config{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};
  std::vector<ulong> maxIndices;
  CNN::Tensor3D<double> out = CNN::Pool<double>::propagate(input, config, maxIndices);

  CHECK(out.shape.c == 1 && out.shape.h == 2 && out.shape.w == 2, "maxpool shape");
  CHECK_NEAR(out.at(0, 0, 0), 6.0, 1e-9, "maxpool top-left");
  CHECK_NEAR(out.at(0, 0, 1), 8.0, 1e-9, "maxpool top-right");
  CHECK_NEAR(out.at(0, 1, 0), 14.0, 1e-9, "maxpool bottom-left");
  CHECK_NEAR(out.at(0, 1, 1), 16.0, 1e-9, "maxpool bottom-right");

  CNN::Tensor3D<double> dOut({1, 2, 2}, 1.0);
  CNN::Tensor3D<double> dIn = CNN::Pool<double>::backpropagate(dOut, input.shape, config, maxIndices);
  CHECK(dIn.shape.h == 4 && dIn.shape.w == 4, "maxpool backprop shape");
  CHECK_NEAR(dIn.at(0, 1, 1), 1.0, 1e-9, "maxpool backprop max pos");
  CHECK_NEAR(dIn.at(0, 0, 0), 0.0, 1e-9, "maxpool backprop non-max");
}

//===================================================================================================================//

static void testAvgPool()
{
  std::cout << "--- testAvgPool ---" << std::endl;

  CNN::Tensor3D<double> input({1, 4, 4});

  for (ulong i = 0; i < 16; i++)
    input.data[i] = static_cast<double>(i + 1);

  CNN::PoolLayerConfig config{CNN::PoolTypeEnum::AVG, 2, 2, 2, 2};
  std::vector<ulong> maxIndices;
  CNN::Tensor3D<double> out = CNN::Pool<double>::propagate(input, config, maxIndices);

  CHECK(out.shape.c == 1 && out.shape.h == 2 && out.shape.w == 2, "avgpool shape");
  CHECK(maxIndices.empty(), "avgpool no maxIndices");
  // Top-left: (1+2+5+6)/4 = 14/4 = 3.5
  CHECK_NEAR(out.at(0, 0, 0), 3.5, 1e-9, "avgpool top-left");
  // Bottom-right: (11+12+15+16)/4 = 54/4 = 13.5
  CHECK_NEAR(out.at(0, 1, 1), 13.5, 1e-9, "avgpool bottom-right");

  // Backprop: gradient distributed evenly
  CNN::Tensor3D<double> dOut({1, 2, 2}, 4.0); // gradient = 4.0
  CNN::Tensor3D<double> dIn = CNN::Pool<double>::backpropagate(dOut, input.shape, config, maxIndices);
  // Each element gets 4.0 / (2*2) = 1.0
  CHECK_NEAR(dIn.at(0, 0, 0), 1.0, 1e-9, "avgpool backprop");
  CHECK_NEAR(dIn.at(0, 1, 1), 1.0, 1e-9, "avgpool backprop center");
}

//===================================================================================================================//

static void testPoolNonSquare()
{
  std::cout << "--- testPoolNonSquare ---" << std::endl;

  // 1x4x6 input, pool 2x3 stride 2x3 → 1x2x2
  CNN::Tensor3D<double> input({1, 4, 6});

  for (ulong i = 0; i < 24; i++)
    input.data[i] = static_cast<double>(i + 1);
  // Row 0: 1  2  3  4  5  6
  // Row 1: 7  8  9  10 11 12
  // Row 2: 13 14 15 16 17 18
  // Row 3: 19 20 21 22 23 24

  CNN::PoolLayerConfig config{CNN::PoolTypeEnum::MAX, 2, 3, 2, 3};
  std::vector<ulong> maxIndices;
  CNN::Tensor3D<double> out = CNN::Pool<double>::propagate(input, config, maxIndices);

  CHECK(out.shape.c == 1 && out.shape.h == 2 && out.shape.w == 2, "non-square pool shape");
  // (oh=0,ow=0): rows 0-1, cols 0-2 → max(1,2,3,7,8,9) = 9
  CHECK_NEAR(out.at(0, 0, 0), 9.0, 1e-9, "non-square pool [0,0]");
  // (oh=0,ow=1): rows 0-1, cols 3-5 → max(4,5,6,10,11,12) = 12
  CHECK_NEAR(out.at(0, 0, 1), 12.0, 1e-9, "non-square pool [0,1]");
  // (oh=1,ow=1): rows 2-3, cols 3-5 → max(16,17,18,22,23,24) = 24
  CHECK_NEAR(out.at(0, 1, 1), 24.0, 1e-9, "non-square pool [1,1]");
}

//===================================================================================================================//

static void testFlatten()
{
  std::cout << "--- testFlatten ---" << std::endl;

  CNN::Tensor3D<double> input({2, 3, 4});

  for (ulong i = 0; i < 24; i++)
    input.data[i] = static_cast<double>(i);

  CNN::Tensor1D<double> flat = CNN::Flatten<double>::propagate(input);
  CHECK(flat.size() == 24, "flatten size");
  CHECK_NEAR(flat[0], 0.0, 1e-9, "flatten first");
  CHECK_NEAR(flat[23], 23.0, 1e-9, "flatten last");

  CNN::Tensor3D<double> back = CNN::Flatten<double>::backpropagate(flat, input.shape);
  CHECK(back.shape.c == 2 && back.shape.h == 3 && back.shape.w == 4, "unflatten shape");
  CHECK_NEAR(back.at(1, 2, 3), 23.0, 1e-9, "unflatten value");
}

//===================================================================================================================//

static void testInstanceNormInference()
{
  std::cout << "--- testInstanceNormInference ---" << std::endl;

  // 2 channels, 2x2 spatial
  CNN::Shape3D shape{2, 2, 2};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0, // channel 0
                5.0, 6.0, 7.0, 8.0}; // channel 1

  CNN::NormParameters<double> params;
  params.numChannels = 2;
  params.gamma = {2.0, 0.5};
  params.beta = {1.0, -1.0};
  params.runningMean = {2.5, 6.5};
  params.runningVar = {1.25, 1.25};

  CNN::NormLayerConfig config;
  config.epsilon = 0.0; // zero eps for exact math

  std::vector<CNN::Tensor3D<double>*> batch = {&input};
  CNN::Normalization<double>::propagate(batch, shape, params, config, CNN::LayerType::INSTANCENORM, false);

  CHECK(input.shape.c == 2 && input.shape.h == 2 && input.shape.w == 2, "instancenorm inference shape");

  // InstanceNorm inference computes per-sample spatial stats (not running stats)
  // channel 0: mean=(1+2+3+4)/4=2.5, var=1.25, invStd = 1/sqrt(1.25)
  double invStd0 = 1.0 / std::sqrt(1.25);

  CHECK_NEAR(input.data[0], 2.0 * (1.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [0]");
  CHECK_NEAR(input.data[1], 2.0 * (2.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [1]");
  CHECK_NEAR(input.data[2], 2.0 * (3.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [2]");
  CHECK_NEAR(input.data[3], 2.0 * (4.0 - 2.5) * invStd0 + 1.0, 1e-9, "instancenorm infer ch0 [3]");

  // channel 1: same invStd
  CHECK_NEAR(input.data[4], 0.5 * (5.0 - 6.5) * invStd0 - 1.0, 1e-9, "instancenorm infer ch1 [0]");
  CHECK_NEAR(input.data[7], 0.5 * (8.0 - 6.5) * invStd0 - 1.0, 1e-9, "instancenorm infer ch1 [3]");

  // Running stats should not change during inference
  CHECK_NEAR(params.runningMean[0], 2.5, 1e-9, "instancenorm infer runningMean unchanged");
  CHECK_NEAR(params.runningVar[0], 1.25, 1e-9, "instancenorm infer runningVar unchanged");
}

//===================================================================================================================//

static void testInstanceNormTraining()
{
  std::cout << "--- testInstanceNormTraining ---" << std::endl;

  // 2 channels, 2x2 spatial
  CNN::Shape3D shape{2, 2, 2};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0, // channel 0
                5.0, 6.0, 7.0, 8.0}; // channel 1

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

  // channel 0: mean = (1+2+3+4)/4 = 2.5, var = ((−1.5)²+(−0.5)²+(0.5)²+(1.5)²)/4 = 1.25
  // Stats indexed as n*C+c, with N=1: statsMean[0*2+0]=2.5, statsMean[0*2+1]=6.5
  CHECK_NEAR(statsMean[0], 2.5, 1e-9, "instancenorm train mean ch0");
  CHECK_NEAR(statsVar[0], 1.25, 1e-9, "instancenorm train var ch0");

  // channel 1: mean = (5+6+7+8)/4 = 6.5, var = 1.25
  CHECK_NEAR(statsMean[1], 6.5, 1e-9, "instancenorm train mean ch1");
  CHECK_NEAR(statsVar[1], 1.25, 1e-9, "instancenorm train var ch1");

  // With gamma=1, beta=0: output = xNormalized
  double invStd = 1.0 / std::sqrt(1.25);
  CHECK_NEAR(input.data[0], (1.0 - 2.5) * invStd, 1e-9, "instancenorm train out ch0 [0]");
  CHECK_NEAR(input.data[3], (4.0 - 2.5) * invStd, 1e-9, "instancenorm train out ch0 [3]");

  // xNormalized should match output when gamma=1, beta=0
  CHECK_NEAR(xNorm[0].data[0], input.data[0], 1e-9, "instancenorm train xNorm matches out");
  CHECK_NEAR(xNorm[0].data[7], input.data[7], 1e-9, "instancenorm train xNorm matches out ch1");

  // Running stats updated: runningMean = 0.9*0.0 + 0.1*2.5 = 0.25
  CHECK_NEAR(params.runningMean[0], 0.25, 1e-7, "instancenorm train runningMean updated ch0");
  CHECK_NEAR(params.runningMean[1], 0.65, 1e-7, "instancenorm train runningMean updated ch1");
  // runningVar = 0.9*1.0 + 0.1*1.25 = 1.025
  CHECK_NEAR(params.runningVar[0], 1.025, 1e-7, "instancenorm train runningVar updated ch0");
}

//===================================================================================================================//

static void testInstanceNormBackpropagate()
{
  std::cout << "--- testInstanceNormBackpropagate ---" << std::endl;

  // 1 channel, 1x4 spatial for simple math
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

  // Forward pass (training) to get intermediates
  std::vector<double> statsMean, statsVar;
  std::vector<CNN::Tensor3D<double>> xNorm;

  std::vector<CNN::Tensor3D<double>*> batch = {&input};
  CNN::Normalization<double>::propagate(batch, shape, params, config, CNN::LayerType::INSTANCENORM, true, &xNorm,
                                        &statsMean, &statsVar);

  // mean=2.5, var=1.25
  CHECK_NEAR(statsMean[0], 2.5, 1e-9, "backprop setup mean");
  CHECK_NEAR(statsVar[0], 1.25, 1e-9, "backprop setup var");

  // Upstream gradient: all ones
  CNN::Tensor3D<double> dOutput(shape, 1.0);

  std::vector<double> dGamma, dBeta;
  std::vector<CNN::Tensor3D<double>*> dBatch = {&dOutput};
  CNN::Normalization<double>::backpropagate(dBatch, shape, params, config, CNN::LayerType::INSTANCENORM, statsMean,
                                            statsVar, xNorm, dGamma, dBeta);

  // dBeta = sum(dOutput) = 4.0
  CHECK_NEAR(dBeta[0], 4.0, 1e-9, "instancenorm backprop dBeta");

  // dGamma = sum(dOutput * xNorm) = sum(xNorm) = sum of normalized values = 0 (by construction)
  CHECK_NEAR(dGamma[0], 0.0, 1e-9, "instancenorm backprop dGamma");

  // dOutput shape should match (modified in-place)
  CHECK(dOutput.shape.c == 1 && dOutput.shape.h == 1 && dOutput.shape.w == 4, "instancenorm backprop shape");

  // With uniform dOutput=1 and dGamma=0: dInput should be 0
  for (ulong i = 0; i < 4; i++)
    CHECK_NEAR(dOutput.data[i], 0.0, 1e-9, "instancenorm backprop dInput uniform=0");
}

//===================================================================================================================//

static void testInstanceNormBackpropGradient()
{
  std::cout << "--- testInstanceNormBackpropGradient ---" << std::endl;

  // Non-uniform gradient to test actual gradient flow
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

  // Forward pass
  CNN::Tensor3D<double> input(shape);
  input.data = inputOrig.data;
  std::vector<double> statsMean, statsVar;
  std::vector<CNN::Tensor3D<double>> xNorm;
  std::vector<CNN::Tensor3D<double>*> batch = {&input};
  CNN::Normalization<double>::propagate(batch, shape, params, config, CNN::LayerType::INSTANCENORM, true, &xNorm,
                                        &statsMean, &statsVar);

  // Non-uniform upstream gradient
  CNN::Tensor3D<double> dOutput(shape);
  dOutput.data = {0.1, 0.2, 0.3, 0.4};

  std::vector<double> dGamma, dBeta;
  std::vector<CNN::Tensor3D<double>*> dBatch = {&dOutput};
  CNN::Normalization<double>::backpropagate(dBatch, shape, params, config, CNN::LayerType::INSTANCENORM, statsMean,
                                            statsVar, xNorm, dGamma, dBeta);

  // dBeta = sum(dOutput) = 1.0
  CHECK_NEAR(dBeta[0], 1.0, 1e-9, "grad dBeta");

  // Numerical gradient check: perturb each input by eps and compute finite difference
  double eps = 1e-5;

  for (ulong i = 0; i < 4; i++) {
    // Forward with input[i] + eps
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

    // Forward with input[i] - eps
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

    // Numerical gradient = sum_j dOutput[j] * (outPlus[j] - outMinus[j]) / (2*eps)
    double numGrad = 0.0;

    for (ulong j = 0; j < 4; j++)
      numGrad += 0.1 * (j + 1) * (inputPlus.data[j] - inputMinus.data[j]) / (2.0 * eps);

    CHECK_NEAR(dOutput.data[i], numGrad, 1e-4, "instancenorm numerical gradient check");
  }
}

//===================================================================================================================//

static void testGlobalAvgPoolPropagate()
{
  std::cout << "--- testGlobalAvgPoolPropagate ---" << std::endl;

  // 2 channels, 2x3 spatial
  CNN::Shape3D shape{2, 2, 3};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0, // channel 0: mean = 3.5
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0}; // channel 1: mean = 9.5

  CNN::GlobalAvgPool<double>::propagate(input, shape);

  CHECK(input.shape.c == 2 && input.shape.h == 1 && input.shape.w == 1, "gap output shape");
  CHECK(input.data.size() == 2, "gap output size");
  CHECK_NEAR(input.data[0], 3.5, 1e-9, "gap ch0 mean");
  CHECK_NEAR(input.data[1], 9.5, 1e-9, "gap ch1 mean");
}

//===================================================================================================================//

static void testGlobalAvgPoolBackpropagate()
{
  std::cout << "--- testGlobalAvgPoolBackpropagate ---" << std::endl;

  // Original input was 2 channels, 2x3 spatial
  CNN::Shape3D inputShape{2, 2, 3};
  ulong spatialSize = 6;

  // Gradient output has shape (2, 1, 1)
  CNN::Tensor3D<double> gradOutput({2, 1, 1});
  gradOutput.data = {1.2, 0.6};

  CNN::GlobalAvgPool<double>::backpropagate(gradOutput, inputShape);

  CHECK(gradOutput.shape.c == 2 && gradOutput.shape.h == 2 && gradOutput.shape.w == 3, "gap backprop shape");
  CHECK(gradOutput.data.size() == 12, "gap backprop size");

  double expected0 = 1.2 / static_cast<double>(spatialSize);
  double expected1 = 0.6 / static_cast<double>(spatialSize);

  for (ulong s = 0; s < spatialSize; s++) {
    CHECK_NEAR(gradOutput.data[s], expected0, 1e-9, "gap backprop ch0 uniform");
    CHECK_NEAR(gradOutput.data[spatialSize + s], expected1, 1e-9, "gap backprop ch1 uniform");
  }
}

//===================================================================================================================//

static void testGlobalAvgPoolGradientCheck()
{
  std::cout << "--- testGlobalAvgPoolGradientCheck ---" << std::endl;

  // Numerical gradient check: perturb each input element and verify
  CNN::Shape3D shape{2, 1, 3};
  std::vector<double> origData = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
  double eps = 1e-5;

  // Upstream gradient (one per channel)
  std::vector<double> dOut = {0.7, -0.3};

  // Analytical backward
  CNN::Tensor3D<double> gradOutput({2, 1, 1});
  gradOutput.data = dOut;
  CNN::GlobalAvgPool<double>::backpropagate(gradOutput, shape);

  for (ulong i = 0; i < origData.size(); i++) {
    // Forward with input[i] + eps
    CNN::Tensor3D<double> plus(shape);
    plus.data = origData;
    plus.data[i] += eps;
    CNN::GlobalAvgPool<double>::propagate(plus, shape);

    // Forward with input[i] - eps
    CNN::Tensor3D<double> minus(shape);
    minus.data = origData;
    minus.data[i] -= eps;
    CNN::GlobalAvgPool<double>::propagate(minus, shape);

    // Numerical gradient = sum_c dOut[c] * (plus[c] - minus[c]) / (2*eps)
    double numGrad = 0.0;

    for (ulong c = 0; c < 2; c++)
      numGrad += dOut[c] * (plus.data[c] - minus.data[c]) / (2.0 * eps);

    CHECK_NEAR(gradOutput.data[i], numGrad, 1e-6, "gap numerical gradient check");
  }
}

//===================================================================================================================//

static void testGlobalAvgPoolSingleChannel()
{
  std::cout << "--- testGlobalAvgPoolSingleChannel ---" << std::endl;

  CNN::Shape3D shape{1, 4, 4};
  CNN::Tensor3D<double> input(shape);

  for (ulong i = 0; i < 16; i++)
    input.data[i] = static_cast<double>(i + 1); // 1..16, mean = 8.5

  CNN::GlobalAvgPool<double>::propagate(input, shape);

  CHECK(input.shape.c == 1 && input.shape.h == 1 && input.shape.w == 1, "gap single ch shape");
  CHECK_NEAR(input.data[0], 8.5, 1e-9, "gap single ch mean");
}

//===================================================================================================================//

static void testGlobalAvgPoolIdentity1x1()
{
  std::cout << "--- testGlobalAvgPoolIdentity1x1 ---" << std::endl;

  // 1x1 spatial should be identity
  CNN::Shape3D shape{3, 1, 1};
  CNN::Tensor3D<double> input(shape);
  input.data = {2.5, -1.0, 7.3};

  CNN::GlobalAvgPool<double>::propagate(input, shape);

  CHECK(input.shape.c == 3 && input.shape.h == 1 && input.shape.w == 1, "gap 1x1 shape");
  CHECK_NEAR(input.data[0], 2.5, 1e-9, "gap 1x1 ch0");
  CHECK_NEAR(input.data[1], -1.0, 1e-9, "gap 1x1 ch1");
  CHECK_NEAR(input.data[2], 7.3, 1e-9, "gap 1x1 ch2");

  // Backprop on 1x1 should also be identity
  CNN::Tensor3D<double> grad({3, 1, 1});
  grad.data = {0.1, 0.2, 0.3};

  CNN::GlobalAvgPool<double>::backpropagate(grad, shape);

  CHECK(grad.shape.c == 3 && grad.shape.h == 1 && grad.shape.w == 1, "gap 1x1 backprop shape");
  CHECK_NEAR(grad.data[0], 0.1, 1e-9, "gap 1x1 backprop ch0");
  CHECK_NEAR(grad.data[1], 0.2, 1e-9, "gap 1x1 backprop ch1");
  CHECK_NEAR(grad.data[2], 0.3, 1e-9, "gap 1x1 backprop ch2");
}

//===================================================================================================================//

static void testGlobalAvgPoolUniformInput()
{
  std::cout << "--- testGlobalAvgPoolUniformInput ---" << std::endl;

  // Uniform input: mean should equal the uniform value
  CNN::Shape3D shape{2, 3, 3};
  CNN::Tensor3D<double> input(shape, 5.0);

  CNN::GlobalAvgPool<double>::propagate(input, shape);

  CHECK_NEAR(input.data[0], 5.0, 1e-9, "gap uniform ch0");
  CHECK_NEAR(input.data[1], 5.0, 1e-9, "gap uniform ch1");
}

//===================================================================================================================//

static void testGlobalAvgPoolLargeSpatial()
{
  std::cout << "--- testGlobalAvgPoolLargeSpatial ---" << std::endl;

  // Large spatial: 4 channels, 32x32
  CNN::Shape3D shape{4, 32, 32};
  CNN::Tensor3D<double> input(shape);
  ulong spatialSize = 32 * 32;

  for (ulong c = 0; c < 4; c++)

    for (ulong s = 0; s < spatialSize; s++)
      input.data[c * spatialSize + s] = static_cast<double>(c * 100 + s);

  // Expected mean for channel c: c*100 + (spatialSize-1)/2
  CNN::GlobalAvgPool<double>::propagate(input, shape);

  CHECK(input.data.size() == 4, "gap large size");

  for (ulong c = 0; c < 4; c++) {
    double expected = static_cast<double>(c * 100) + static_cast<double>(spatialSize - 1) / 2.0;
    CHECK_NEAR(input.data[c], expected, 1e-6, "gap large mean");
  }
}

//===================================================================================================================//

static void testGlobalAvgPoolMultiChannelGradient()
{
  std::cout << "--- testGlobalAvgPoolMultiChannelGradient ---" << std::endl;

  // 4 channels, 3x3 spatial — full numerical gradient check
  CNN::Shape3D shape{4, 3, 3};
  ulong total = 4 * 3 * 3;
  std::vector<double> origData(total);

  for (ulong i = 0; i < total; i++)
    origData[i] = static_cast<double>(i) * 0.1 - 1.5;

  std::vector<double> dOut = {0.5, -0.3, 1.0, -0.7};
  double eps = 1e-5;

  // Analytical backward
  CNN::Tensor3D<double> gradOutput({4, 1, 1});
  gradOutput.data = dOut;
  CNN::GlobalAvgPool<double>::backpropagate(gradOutput, shape);

  // Numerical check
  for (ulong i = 0; i < total; i++) {
    CNN::Tensor3D<double> plus(shape);
    plus.data = origData;
    plus.data[i] += eps;
    CNN::GlobalAvgPool<double>::propagate(plus, shape);

    CNN::Tensor3D<double> minus(shape);
    minus.data = origData;
    minus.data[i] -= eps;
    CNN::GlobalAvgPool<double>::propagate(minus, shape);

    double numGrad = 0.0;

    for (ulong c = 0; c < 4; c++)
      numGrad += dOut[c] * (plus.data[c] - minus.data[c]) / (2.0 * eps);

    CHECK_NEAR(gradOutput.data[i], numGrad, 1e-6, "gap multi-ch gradient check");
  }
}

//===================================================================================================================//

static void testGlobalDualPoolPropagate()
{
  std::cout << "--- testGlobalDualPoolPropagate ---" << std::endl;

  // 2 channels, 2x3 spatial
  CNN::Shape3D shape{2, 2, 3};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0, // ch0: avg=3.5, max=6
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0}; // ch1: avg=9.5, max=12

  CNN::GlobalDualPool<double>::propagate(input, shape);

  CHECK(input.shape.c == 4 && input.shape.h == 1 && input.shape.w == 1, "gdp output shape");
  CHECK(input.data.size() == 4, "gdp output size");
  CHECK_NEAR(input.data[0], 3.5, 1e-9, "gdp ch0 avg");
  CHECK_NEAR(input.data[1], 9.5, 1e-9, "gdp ch1 avg");
  CHECK_NEAR(input.data[2], 6.0, 1e-9, "gdp ch0 max");
  CHECK_NEAR(input.data[3], 12.0, 1e-9, "gdp ch1 max");
}

//===================================================================================================================//

static void testGlobalDualPoolBackpropagate()
{
  std::cout << "--- testGlobalDualPoolBackpropagate ---" << std::endl;

  // Original input: 2 channels, 2x3 spatial
  CNN::Shape3D inputShape{2, 2, 3};
  ulong spatialSize = 6;

  CNN::Tensor3D<double> layerInput(inputShape);
  layerInput.data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // ch0: max at index 5
                     7.0, 8.0, 9.0, 10.0, 11.0, 12.0}; // ch1: max at index 5

  // Gradient output: (4, 1, 1) — [dAvg0, dAvg1, dMax0, dMax1]
  CNN::Tensor3D<double> gradOutput({4, 1, 1});
  gradOutput.data = {1.2, 0.6, 0.5, 0.3};

  CNN::GlobalDualPool<double>::backpropagate(gradOutput, layerInput, inputShape);

  CHECK(gradOutput.shape.c == 2 && gradOutput.shape.h == 2 && gradOutput.shape.w == 3, "gdp backprop shape");
  CHECK(gradOutput.data.size() == 12, "gdp backprop size");

  double avgGrad0 = 1.2 / static_cast<double>(spatialSize);
  double avgGrad1 = 0.6 / static_cast<double>(spatialSize);

  // ch0: avg gradient + max gradient at index 5 only
  for (ulong s = 0; s < spatialSize; s++) {
    double expected = avgGrad0 + (s == 5 ? 0.5 : 0.0);
    CHECK_NEAR(gradOutput.data[s], expected, 1e-9, "gdp backprop ch0");
  }

  // ch1: avg gradient + max gradient at index 5 only
  for (ulong s = 0; s < spatialSize; s++) {
    double expected = avgGrad1 + (s == 5 ? 0.3 : 0.0);
    CHECK_NEAR(gradOutput.data[spatialSize + s], expected, 1e-9, "gdp backprop ch1");
  }
}

//===================================================================================================================//

static void testGlobalDualPoolSingleSpatial()
{
  std::cout << "--- testGlobalDualPoolSingleSpatial ---" << std::endl;

  // (4, 1, 1) -> (8, 1, 1), avg == max for each channel
  CNN::Shape3D shape{4, 1, 1};
  CNN::Tensor3D<double> input(shape);
  input.data = {2.5, -1.0, 7.3, 0.0};

  CNN::GlobalDualPool<double>::propagate(input, shape);

  CHECK(input.shape.c == 8 && input.shape.h == 1 && input.shape.w == 1, "gdp 1x1 shape");

  for (ulong c = 0; c < 4; c++) {
    CHECK_NEAR(input.data[c], input.data[4 + c], 1e-9, "gdp 1x1 avg==max");
  }
}

//===================================================================================================================//

static void testGlobalDualPoolNegativeValues()
{
  std::cout << "--- testGlobalDualPoolNegativeValues ---" << std::endl;

  CNN::Shape3D shape{1, 2, 2};
  CNN::Tensor3D<double> input(shape);
  input.data = {-5.0, -3.0, -8.0, -1.0}; // avg = -4.25, max = -1.0

  CNN::GlobalDualPool<double>::propagate(input, shape);

  CHECK_NEAR(input.data[0], -4.25, 1e-9, "gdp neg avg");
  CHECK_NEAR(input.data[1], -1.0, 1e-9, "gdp neg max");
}

//===================================================================================================================//

static void testGlobalDualPoolUniformChannel()
{
  std::cout << "--- testGlobalDualPoolUniformChannel ---" << std::endl;

  CNN::Shape3D shape{2, 3, 3};
  CNN::Tensor3D<double> input(shape, 5.0);

  CNN::GlobalDualPool<double>::propagate(input, shape);

  CHECK_NEAR(input.data[0], 5.0, 1e-9, "gdp uniform avg ch0");
  CHECK_NEAR(input.data[1], 5.0, 1e-9, "gdp uniform avg ch1");
  CHECK_NEAR(input.data[2], 5.0, 1e-9, "gdp uniform max ch0");
  CHECK_NEAR(input.data[3], 5.0, 1e-9, "gdp uniform max ch1");
}

//===================================================================================================================//

static void testGlobalDualPoolLargeSpatial()
{
  std::cout << "--- testGlobalDualPoolLargeSpatial ---" << std::endl;

  CNN::Shape3D shape{1, 100, 100};
  CNN::Tensor3D<double> input(shape);
  ulong spatialSize = 10000;

  for (ulong s = 0; s < spatialSize; s++)
    input.data[s] = static_cast<double>(s);

  // avg = (0 + 9999) / 2 = 4999.5, max = 9999
  CNN::GlobalDualPool<double>::propagate(input, shape);

  CHECK(input.data.size() == 2, "gdp large size");
  CHECK_NEAR(input.data[0], 4999.5, 1e-6, "gdp large avg");
  CHECK_NEAR(input.data[1], 9999.0, 1e-9, "gdp large max");
}

//===================================================================================================================//

static void testGlobalDualPoolGradientCheck()
{
  std::cout << "--- testGlobalDualPoolGradientCheck ---" << std::endl;

  CNN::Shape3D shape{2, 3, 3};
  ulong total = 2 * 3 * 3;
  std::vector<double> origData(total);

  for (ulong i = 0; i < total; i++)
    origData[i] = static_cast<double>(i) * 0.1 - 0.5;

  // dOutput has 4 elements: [dAvg0, dAvg1, dMax0, dMax1]
  std::vector<double> dOut = {0.5, -0.3, 0.7, -0.2};
  double eps = 1e-5;

  // Analytical backward
  CNN::Tensor3D<double> layerInput(shape);
  layerInput.data = origData;

  CNN::Tensor3D<double> gradOutput({4, 1, 1});
  gradOutput.data = dOut;
  CNN::GlobalDualPool<double>::backpropagate(gradOutput, layerInput, shape);

  // Numerical check
  for (ulong i = 0; i < total; i++) {
    CNN::Tensor3D<double> plus(shape);
    plus.data = origData;
    plus.data[i] += eps;
    CNN::GlobalDualPool<double>::propagate(plus, shape);

    CNN::Tensor3D<double> minus(shape);
    minus.data = origData;
    minus.data[i] -= eps;
    CNN::GlobalDualPool<double>::propagate(minus, shape);

    double numGrad = 0.0;

    for (ulong c = 0; c < 4; c++)
      numGrad += dOut[c] * (plus.data[c] - minus.data[c]) / (2.0 * eps);

    CHECK_NEAR(gradOutput.data[i], numGrad, 1e-6, "gdp numerical gradient check");
  }
}

//===================================================================================================================//

static void testGlobalDualPoolAvgOnlyGradient()
{
  std::cout << "--- testGlobalDualPoolAvgOnlyGradient ---" << std::endl;

  CNN::Shape3D shape{1, 2, 2};
  CNN::Tensor3D<double> layerInput(shape);
  layerInput.data = {1.0, 3.0, 2.0, 4.0}; // max at index 3

  // Zero max gradient — only avg gradient
  CNN::Tensor3D<double> gradOutput({2, 1, 1});
  gradOutput.data = {1.0, 0.0}; // dAvg=1.0, dMax=0.0

  CNN::GlobalDualPool<double>::backpropagate(gradOutput, layerInput, shape);

  double expected = 1.0 / 4.0;

  for (ulong s = 0; s < 4; s++)
    CHECK_NEAR(gradOutput.data[s], expected, 1e-9, "gdp avg-only gradient");
}

//===================================================================================================================//

static void testGlobalDualPoolMaxOnlyGradient()
{
  std::cout << "--- testGlobalDualPoolMaxOnlyGradient ---" << std::endl;

  CNN::Shape3D shape{1, 2, 2};
  CNN::Tensor3D<double> layerInput(shape);
  layerInput.data = {1.0, 3.0, 2.0, 4.0}; // max at index 3

  // Zero avg gradient — only max gradient
  CNN::Tensor3D<double> gradOutput({2, 1, 1});
  gradOutput.data = {0.0, 2.5}; // dAvg=0.0, dMax=2.5

  CNN::GlobalDualPool<double>::backpropagate(gradOutput, layerInput, shape);

  for (ulong s = 0; s < 4; s++) {
    double expected = (s == 3) ? 2.5 : 0.0;
    CHECK_NEAR(gradOutput.data[s], expected, 1e-9, "gdp max-only gradient");
  }
}

//===================================================================================================================//

void runLayerTests()
{
  testTensor3D();
  testReLU();
  testMaxPool();
  testAvgPool();
  testPoolNonSquare();
  testFlatten();
  testGlobalAvgPoolPropagate();
  testGlobalAvgPoolBackpropagate();
  testGlobalAvgPoolGradientCheck();
  testGlobalAvgPoolSingleChannel();
  testGlobalAvgPoolIdentity1x1();
  testGlobalAvgPoolUniformInput();
  testGlobalAvgPoolLargeSpatial();
  testGlobalAvgPoolMultiChannelGradient();
  testGlobalDualPoolPropagate();
  testGlobalDualPoolBackpropagate();
  testGlobalDualPoolSingleSpatial();
  testGlobalDualPoolNegativeValues();
  testGlobalDualPoolUniformChannel();
  testGlobalDualPoolLargeSpatial();
  testGlobalDualPoolGradientCheck();
  testGlobalDualPoolAvgOnlyGradient();
  testGlobalDualPoolMaxOnlyGradient();
  testInstanceNormInference();
  testInstanceNormTraining();
  testInstanceNormBackpropagate();
  testInstanceNormBackpropGradient();
}
