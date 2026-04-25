#include "test_helpers.hpp"

#include "CNN_GlobalAvgPool.hpp"

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

  CNN::Shape3D inputShape{2, 2, 3};
  ulong spatialSize = 6;

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

  CNN::Shape3D shape{2, 1, 3};
  std::vector<double> origData = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
  double eps = 1e-5;

  std::vector<double> dOut = {0.7, -0.3};

  CNN::Tensor3D<double> gradOutput({2, 1, 1});
  gradOutput.data = dOut;
  CNN::GlobalAvgPool<double>::backpropagate(gradOutput, shape);

  for (ulong i = 0; i < origData.size(); i++) {
    CNN::Tensor3D<double> plus(shape);
    plus.data = origData;
    plus.data[i] += eps;
    CNN::GlobalAvgPool<double>::propagate(plus, shape);

    CNN::Tensor3D<double> minus(shape);
    minus.data = origData;
    minus.data[i] -= eps;
    CNN::GlobalAvgPool<double>::propagate(minus, shape);

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
    input.data[i] = static_cast<double>(i + 1);

  CNN::GlobalAvgPool<double>::propagate(input, shape);

  CHECK(input.shape.c == 1 && input.shape.h == 1 && input.shape.w == 1, "gap single ch shape");
  CHECK_NEAR(input.data[0], 8.5, 1e-9, "gap single ch mean");
}

//===================================================================================================================//

static void testGlobalAvgPoolIdentity1x1()
{
  std::cout << "--- testGlobalAvgPoolIdentity1x1 ---" << std::endl;

  CNN::Shape3D shape{3, 1, 1};
  CNN::Tensor3D<double> input(shape);
  input.data = {2.5, -1.0, 7.3};

  CNN::GlobalAvgPool<double>::propagate(input, shape);

  CHECK(input.shape.c == 3 && input.shape.h == 1 && input.shape.w == 1, "gap 1x1 shape");
  CHECK_NEAR(input.data[0], 2.5, 1e-9, "gap 1x1 ch0");
  CHECK_NEAR(input.data[1], -1.0, 1e-9, "gap 1x1 ch1");
  CHECK_NEAR(input.data[2], 7.3, 1e-9, "gap 1x1 ch2");

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

  CNN::Shape3D shape{4, 32, 32};
  CNN::Tensor3D<double> input(shape);
  ulong spatialSize = 32 * 32;

  for (ulong c = 0; c < 4; c++)

    for (ulong s = 0; s < spatialSize; s++)
      input.data[c * spatialSize + s] = static_cast<double>(c * 100 + s);

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

  CNN::Shape3D shape{4, 3, 3};
  ulong total = 4 * 3 * 3;
  std::vector<double> origData(total);

  for (ulong i = 0; i < total; i++)
    origData[i] = static_cast<double>(i) * 0.1 - 1.5;

  std::vector<double> dOut = {0.5, -0.3, 1.0, -0.7};
  double eps = 1e-5;

  CNN::Tensor3D<double> gradOutput({4, 1, 1});
  gradOutput.data = dOut;
  CNN::GlobalAvgPool<double>::backpropagate(gradOutput, shape);

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

void runGlobalAvgPoolTests()
{
  testGlobalAvgPoolPropagate();
  testGlobalAvgPoolBackpropagate();
  testGlobalAvgPoolGradientCheck();
  testGlobalAvgPoolSingleChannel();
  testGlobalAvgPoolIdentity1x1();
  testGlobalAvgPoolUniformInput();
  testGlobalAvgPoolLargeSpatial();
  testGlobalAvgPoolMultiChannelGradient();
}