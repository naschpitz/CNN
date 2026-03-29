#include "test_helpers.hpp"

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

void runLayerTests()
{
  testTensor3D();
  testReLU();
  testMaxPool();
  testAvgPool();
  testPoolNonSquare();
  testFlatten();
}
