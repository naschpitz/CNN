#include "test_helpers.hpp"

#include "CNN_GlobalDualPool.hpp"

//===================================================================================================================//

static void testGlobalDualPoolPropagate()
{
  std::cout << "--- testGlobalDualPoolPropagate ---" << std::endl;

  CNN::Shape3D shape{2, 2, 3};
  CNN::Tensor3D<double> input(shape);
  input.data = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

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

  CNN::Shape3D inputShape{2, 2, 3};
  ulong spatialSize = 6;

  CNN::Tensor3D<double> layerInput(inputShape);
  layerInput.data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                     7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

  CNN::Tensor3D<double> gradOutput({4, 1, 1});
  gradOutput.data = {1.2, 0.6, 0.5, 0.3};

  CNN::GlobalDualPool<double>::backpropagate(gradOutput, layerInput, inputShape);

  CHECK(gradOutput.shape.c == 2 && gradOutput.shape.h == 2 && gradOutput.shape.w == 3, "gdp backprop shape");
  CHECK(gradOutput.data.size() == 12, "gdp backprop size");

  double avgGrad0 = 1.2 / static_cast<double>(spatialSize);
  double avgGrad1 = 0.6 / static_cast<double>(spatialSize);

  for (ulong s = 0; s < spatialSize; s++) {
    double expected = avgGrad0 + (s == 5 ? 0.5 : 0.0);
    CHECK_NEAR(gradOutput.data[s], expected, 1e-9, "gdp backprop ch0");
  }

  for (ulong s = 0; s < spatialSize; s++) {
    double expected = avgGrad1 + (s == 5 ? 0.3 : 0.0);
    CHECK_NEAR(gradOutput.data[spatialSize + s], expected, 1e-9, "gdp backprop ch1");
  }
}

//===================================================================================================================//

static void testGlobalDualPoolSingleSpatial()
{
  std::cout << "--- testGlobalDualPoolSingleSpatial ---" << std::endl;

  CNN::Shape3D shape{4, 1, 1};
  CNN::Tensor3D<double> input(shape);
  input.data = {2.5, -1.0, 7.3, 0.0};

  CNN::GlobalDualPool<double>::propagate(input, shape);

  CHECK(input.shape.c == 8 && input.shape.h == 1 && input.shape.w == 1, "gdp 1x1 shape");

  for (ulong c = 0; c < 4; c++)
    CHECK_NEAR(input.data[c], input.data[4 + c], 1e-9, "gdp 1x1 avg==max");
}

//===================================================================================================================//

static void testGlobalDualPoolNegativeValues()
{
  std::cout << "--- testGlobalDualPoolNegativeValues ---" << std::endl;

  CNN::Shape3D shape{1, 2, 2};
  CNN::Tensor3D<double> input(shape);
  input.data = {-5.0, -3.0, -8.0, -1.0};

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

  std::vector<double> dOut = {0.5, -0.3, 0.7, -0.2};
  double eps = 1e-5;

  CNN::Tensor3D<double> layerInput(shape);
  layerInput.data = origData;

  CNN::Tensor3D<double> gradOutput({4, 1, 1});
  gradOutput.data = dOut;
  CNN::GlobalDualPool<double>::backpropagate(gradOutput, layerInput, shape);

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
  layerInput.data = {1.0, 3.0, 2.0, 4.0};

  CNN::Tensor3D<double> gradOutput({2, 1, 1});
  gradOutput.data = {1.0, 0.0};

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
  layerInput.data = {1.0, 3.0, 2.0, 4.0};

  CNN::Tensor3D<double> gradOutput({2, 1, 1});
  gradOutput.data = {0.0, 2.5};

  CNN::GlobalDualPool<double>::backpropagate(gradOutput, layerInput, shape);

  for (ulong s = 0; s < 4; s++) {
    double expected = (s == 3) ? 2.5 : 0.0;
    CHECK_NEAR(gradOutput.data[s], expected, 1e-9, "gdp max-only gradient");
  }
}

//===================================================================================================================//

void runGlobalDualPoolTests()
{
  testGlobalDualPoolPropagate();
  testGlobalDualPoolBackpropagate();
  testGlobalDualPoolSingleSpatial();
  testGlobalDualPoolNegativeValues();
  testGlobalDualPoolUniformChannel();
  testGlobalDualPoolLargeSpatial();
  testGlobalDualPoolGradientCheck();
  testGlobalDualPoolAvgOnlyGradient();
  testGlobalDualPoolMaxOnlyGradient();
}
