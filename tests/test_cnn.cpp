#include "CNN_Types.hpp"
#include "CNN_Conv2D.hpp"
#include "CNN_ReLU.hpp"
#include "CNN_Pool.hpp"
#include "CNN_Flatten.hpp"
#include "CNN_Core.hpp"
#include "CNN_CoreConfig.hpp"
#include "CNN_Sample.hpp"
#include "CNN_SlidingStrategy.hpp"
#include "CNN_LayersConfig.hpp"

#include <ANN_ActvFunc.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>

static int testsPassed = 0;
static int testsFailed = 0;

#define CHECK(cond, msg) do { \
  if (!(cond)) { \
    std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    testsFailed++; \
  } else { \
    testsPassed++; \
  } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) CHECK(std::fabs((a) - (b)) < (tol), msg)

// Helper: create gradient-filled tensor (values from lo to hi across spatial dims)
// This produces diverse CNN features, avoiding the uniform-input problem where
// all ANN inputs are identical and random weight initialization can stall learning.
template<typename T>
CNN::Tensor3D<T> makeGradientInput(CNN::Shape3D shape, T lo = T(0.5), T hi = T(1.0)) {
  CNN::Tensor3D<T> t(shape);
  ulong total = shape.c * shape.h * shape.w;
  for (ulong c = 0; c < shape.c; ++c)
    for (ulong h = 0; h < shape.h; ++h)
      for (ulong w = 0; w < shape.w; ++w) {
        ulong idx = c * shape.h * shape.w + h * shape.w + w;
        t.at(c, h, w) = lo + (hi - lo) * T(idx) / T(total > 1 ? total - 1 : 1);
      }
  return t;
}

//===================================================================================================================//

void testTensor3D() {
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

void testConv2DForward() {
  std::cout << "--- testConv2DForward ---" << std::endl;

  CNN::Tensor3D<double> input({1, 4, 4});
  for (ulong i = 0; i < 16; i++) input.data[i] = static_cast<double>(i + 1);

  CNN::ConvLayerConfig config{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};
  CNN::ConvParameters<double> params;
  params.numFilters = 1; params.inputC = 1; params.filterH = 3; params.filterW = 3;
  params.filters.assign(9, 1.0);
  params.biases.assign(1, 0.0);

  CNN::Tensor3D<double> out = CNN::Conv2D<double>::predict(input, config, params);
  CHECK(out.shape.c == 1 && out.shape.h == 2 && out.shape.w == 2, "conv2d output shape");
  CHECK_NEAR(out.at(0, 0, 0), 54.0, 1e-9, "conv2d top-left");
  CHECK_NEAR(out.at(0, 0, 1), 63.0, 1e-9, "conv2d top-right");
  CHECK_NEAR(out.at(0, 1, 0), 90.0, 1e-9, "conv2d bottom-left");
  CHECK_NEAR(out.at(0, 1, 1), 99.0, 1e-9, "conv2d bottom-right");
}

//===================================================================================================================//

void testConv2DBackprop() {
  std::cout << "--- testConv2DBackprop ---" << std::endl;

  CNN::Tensor3D<double> input({1, 4, 4});
  for (ulong i = 0; i < 16; i++) input.data[i] = static_cast<double>(i + 1);

  CNN::ConvLayerConfig config{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};
  CNN::ConvParameters<double> params;
  params.numFilters = 1; params.inputC = 1; params.filterH = 3; params.filterW = 3;
  params.filters.assign(9, 1.0);
  params.biases.assign(1, 0.0);

  CNN::Tensor3D<double> dOut({1, 2, 2}, 1.0);
  std::vector<double> dFilters, dBiases;
  CNN::Tensor3D<double> dInput = CNN::Conv2D<double>::backpropagate(dOut, input, config, params, dFilters, dBiases);

  CHECK(dInput.shape.c == 1 && dInput.shape.h == 4 && dInput.shape.w == 4, "backprop dInput shape");
  CHECK(dFilters.size() == 9, "backprop dFilters size");
  CHECK(dBiases.size() == 1, "backprop dBiases size");
  CHECK_NEAR(dBiases[0], 4.0, 1e-9, "backprop dBias value");
}

//===================================================================================================================//

void testConv2DWithBias() {
  std::cout << "--- testConv2DWithBias ---" << std::endl;

  // 1x3x3 input, 1 filter 2x2, VALID, bias = 10.0
  CNN::Tensor3D<double> input({1, 3, 3});
  for (ulong i = 0; i < 9; i++) input.data[i] = 1.0;  // all ones

  CNN::ConvLayerConfig config{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};
  CNN::ConvParameters<double> params;
  params.numFilters = 1; params.inputC = 1; params.filterH = 2; params.filterW = 2;
  params.filters.assign(4, 1.0);  // all ones
  params.biases = {10.0};

  CNN::Tensor3D<double> out = CNN::Conv2D<double>::predict(input, config, params);
  // Each output = sum(1*1 for 4 elements) + 10 = 4 + 10 = 14
  CHECK(out.shape.c == 1 && out.shape.h == 2 && out.shape.w == 2, "bias output shape");
  CHECK_NEAR(out.at(0, 0, 0), 14.0, 1e-9, "bias top-left");
  CHECK_NEAR(out.at(0, 1, 1), 14.0, 1e-9, "bias bottom-right");
}

//===================================================================================================================//

void testConv2DMultiFilter() {
  std::cout << "--- testConv2DMultiFilter ---" << std::endl;

  // 1x3x3 input, 2 filters 2x2, VALID
  CNN::Tensor3D<double> input({1, 3, 3});
  for (ulong i = 0; i < 9; i++) input.data[i] = static_cast<double>(i + 1);
  // input: [[1,2,3],[4,5,6],[7,8,9]]

  CNN::ConvLayerConfig config{2, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};
  CNN::ConvParameters<double> params;
  params.numFilters = 2; params.inputC = 1; params.filterH = 2; params.filterW = 2;
  // Filter 0: all 1s, Filter 1: all -1s
  params.filters = {1, 1, 1, 1, -1, -1, -1, -1};
  params.biases = {0.0, 0.0};

  CNN::Tensor3D<double> out = CNN::Conv2D<double>::predict(input, config, params);
  CHECK(out.shape.c == 2, "multifilter output channels");
  CHECK(out.shape.h == 2, "multifilter output height");

  // Filter 0 (all 1s): top-left = 1+2+4+5 = 12
  CHECK_NEAR(out.at(0, 0, 0), 12.0, 1e-9, "filter0 top-left");
  // Filter 0: bottom-right = 5+6+8+9 = 28
  CHECK_NEAR(out.at(0, 1, 1), 28.0, 1e-9, "filter0 bottom-right");
  // Filter 1 (all -1s): top-left = -(1+2+4+5) = -12
  CHECK_NEAR(out.at(1, 0, 0), -12.0, 1e-9, "filter1 top-left");
  CHECK_NEAR(out.at(1, 1, 1), -28.0, 1e-9, "filter1 bottom-right");
}

//===================================================================================================================//

void testConv2DMultiChannel() {
  std::cout << "--- testConv2DMultiChannel ---" << std::endl;

  // 2x2x2 input (2 channels), 1 filter 2x2, VALID
  CNN::Tensor3D<double> input({2, 2, 2});
  // Channel 0: [[1,2],[3,4]]  Channel 1: [[5,6],[7,8]]
  input.data = {1, 2, 3, 4, 5, 6, 7, 8};

  CNN::ConvLayerConfig config{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};
  CNN::ConvParameters<double> params;
  params.numFilters = 1; params.inputC = 2; params.filterH = 2; params.filterW = 2;
  // Filter for ch0: [[1,0],[0,1]], Filter for ch1: [[0,1],[1,0]]
  params.filters = {1, 0, 0, 1, 0, 1, 1, 0};
  params.biases = {0.0};

  CNN::Tensor3D<double> out = CNN::Conv2D<double>::predict(input, config, params);
  CHECK(out.shape.c == 1 && out.shape.h == 1 && out.shape.w == 1, "multichannel output shape");
  // ch0 contribution: 1*1 + 2*0 + 3*0 + 4*1 = 5
  // ch1 contribution: 5*0 + 6*1 + 7*1 + 8*0 = 13
  // Total: 5 + 13 = 18
  CHECK_NEAR(out.at(0, 0, 0), 18.0, 1e-9, "multichannel conv value");
}

//===================================================================================================================//

void testConv2DBackpropValues() {
  std::cout << "--- testConv2DBackpropValues ---" << std::endl;

  // 1x3x3 input, 1 filter 2x2, VALID → 1x2x2 output
  CNN::Tensor3D<double> input({1, 3, 3});
  input.data = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  CNN::ConvLayerConfig config{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};
  CNN::ConvParameters<double> params;
  params.numFilters = 1; params.inputC = 1; params.filterH = 2; params.filterW = 2;
  params.filters = {1, 2, 3, 4};
  params.biases = {0.0};

  CNN::Tensor3D<double> dOut({1, 2, 2});
  dOut.data = {1, 1, 1, 1};  // uniform gradient

  std::vector<double> dFilters, dBiases;
  CNN::Tensor3D<double> dInput = CNN::Conv2D<double>::backpropagate(dOut, input, config, params, dFilters, dBiases);

  // dFilters[0][0][kh][kw] = sum over (oh,ow) of dOut[0][oh][ow] * input[0][oh+kh][ow+kw]
  // dFilters[kh=0,kw=0] = 1*1 + 1*2 + 1*4 + 1*5 = 12
  // dFilters[kh=0,kw=1] = 1*2 + 1*3 + 1*5 + 1*6 = 16
  // dFilters[kh=1,kw=0] = 1*4 + 1*5 + 1*7 + 1*8 = 24
  // dFilters[kh=1,kw=1] = 1*5 + 1*6 + 1*8 + 1*9 = 28
  CHECK_NEAR(dFilters[0], 12.0, 1e-9, "dFilter[0,0]");
  CHECK_NEAR(dFilters[1], 16.0, 1e-9, "dFilter[0,1]");
  CHECK_NEAR(dFilters[2], 24.0, 1e-9, "dFilter[1,0]");
  CHECK_NEAR(dFilters[3], 28.0, 1e-9, "dFilter[1,1]");

  // dInput[0][ih][iw] = sum over (f,oh,ow,kh,kw) of dOut[f][oh][ow] * filter[f][0][kh][kw]
  //   where oh*sY+kh = ih and ow*sX+kw = iw
  // For (ih=0,iw=0): only (oh=0,ow=0,kh=0,kw=0) → 1*1 = 1
  // For (ih=0,iw=1): (oh=0,ow=0,kh=0,kw=1) + (oh=0,ow=1,kh=0,kw=0) → 1*2 + 1*1 = 3
  // For (ih=1,iw=1): all four combos contribute: 1*4 + 1*3 + 1*2 + 1*1 = 10
  CHECK_NEAR(dInput.at(0, 0, 0), 1.0, 1e-9, "dInput[0,0]");
  CHECK_NEAR(dInput.at(0, 0, 1), 3.0, 1e-9, "dInput[0,1]");
  CHECK_NEAR(dInput.at(0, 1, 1), 10.0, 1e-9, "dInput[1,1]");
  // Corner (ih=2,iw=2): only (oh=1,ow=1,kh=1,kw=1) → 1*4 = 4
  CHECK_NEAR(dInput.at(0, 2, 2), 4.0, 1e-9, "dInput[2,2]");
}

//===================================================================================================================//

void testReLU() {
  std::cout << "--- testReLU ---" << std::endl;

  CNN::Tensor3D<double> input({1, 2, 3});
  input.data = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

  CNN::Tensor3D<double> out = CNN::ReLU<double>::predict(input);
  CHECK_NEAR(out.data[0], 0.0, 1e-9, "relu neg -> 0");
  CHECK_NEAR(out.data[3], 1.0, 1e-9, "relu pos passthrough");
  CHECK_NEAR(out.data[5], 3.0, 1e-9, "relu pos passthrough");

  CNN::Tensor3D<double> dOut({1, 2, 3}, 1.0);
  CNN::Tensor3D<double> dIn = CNN::ReLU<double>::backpropagate(dOut, input);
  CHECK_NEAR(dIn.data[0], 0.0, 1e-9, "relu backprop blocks neg");
  CHECK_NEAR(dIn.data[3], 1.0, 1e-9, "relu backprop passes pos");
}

//===================================================================================================================//

void testMaxPool() {
  std::cout << "--- testMaxPool ---" << std::endl;

  CNN::Tensor3D<double> input({1, 4, 4});
  for (ulong i = 0; i < 16; i++) input.data[i] = static_cast<double>(i + 1);

  CNN::PoolLayerConfig config{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};
  std::vector<ulong> maxIndices;
  CNN::Tensor3D<double> out = CNN::Pool<double>::predict(input, config, maxIndices);

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

void testFlatten() {
  std::cout << "--- testFlatten ---" << std::endl;

  CNN::Tensor3D<double> input({2, 3, 4});
  for (ulong i = 0; i < 24; i++) input.data[i] = static_cast<double>(i);

  CNN::Tensor1D<double> flat = CNN::Flatten<double>::predict(input);
  CHECK(flat.size() == 24, "flatten size");
  CHECK_NEAR(flat[0], 0.0, 1e-9, "flatten first");
  CHECK_NEAR(flat[23], 23.0, 1e-9, "flatten last");

  CNN::Tensor3D<double> back = CNN::Flatten<double>::backpropagate(flat, input.shape);
  CHECK(back.shape.c == 2 && back.shape.h == 3 && back.shape.w == 4, "unflatten shape");
  CHECK_NEAR(back.at(1, 2, 3), 23.0, 1e-9, "unflatten value");
}

//===================================================================================================================//

void testConv2DStride() {
  std::cout << "--- testConv2DStride ---" << std::endl;

  // 1x4x4 input, 1 filter 2x2, stride 2, VALID → 1x2x2 output
  CNN::Tensor3D<double> input({1, 4, 4});
  for (ulong i = 0; i < 16; i++) input.data[i] = static_cast<double>(i + 1);

  CNN::ConvLayerConfig config{1, 2, 2, 2, 2, CNN::SlidingStrategyType::VALID};
  CNN::ConvParameters<double> params;
  params.numFilters = 1; params.inputC = 1; params.filterH = 2; params.filterW = 2;
  params.filters.assign(4, 1.0);
  params.biases = {0.0};

  CNN::Tensor3D<double> out = CNN::Conv2D<double>::predict(input, config, params);
  CHECK(out.shape.h == 2 && out.shape.w == 2, "stride output shape");
  // (oh=0,ow=0): sum of input[0:2,0:2] = 1+2+5+6 = 14
  CHECK_NEAR(out.at(0, 0, 0), 14.0, 1e-9, "stride top-left");
  // (oh=0,ow=1): sum of input[0:2,2:4] = 3+4+7+8 = 22
  CHECK_NEAR(out.at(0, 0, 1), 22.0, 1e-9, "stride top-right");
  // (oh=1,ow=0): sum of input[2:4,0:2] = 9+10+13+14 = 46
  CHECK_NEAR(out.at(0, 1, 0), 46.0, 1e-9, "stride bottom-left");
  // (oh=1,ow=1): sum of input[2:4,2:4] = 11+12+15+16 = 54
  CHECK_NEAR(out.at(0, 1, 1), 54.0, 1e-9, "stride bottom-right");
}

//===================================================================================================================//

void testConv2DSamePadding() {
  std::cout << "--- testConv2DSamePadding ---" << std::endl;

  // 1x3x3 input, 1 filter 3x3, stride 1, SAME (pad=1) → 1x3x3 output
  CNN::Tensor3D<double> input({1, 3, 3});
  for (ulong i = 0; i < 9; i++) input.data[i] = 1.0;

  CNN::ConvLayerConfig config{1, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};
  CNN::ConvParameters<double> params;
  params.numFilters = 1; params.inputC = 1; params.filterH = 3; params.filterW = 3;
  params.filters.assign(9, 1.0);
  params.biases = {0.0};

  CNN::Tensor3D<double> out = CNN::Conv2D<double>::predict(input, config, params);
  // SAME with 3x3 kernel → pad = 3/2 = 1. outH = (3+2-3)/1+1 = 3
  CHECK(out.shape.c == 1 && out.shape.h == 3 && out.shape.w == 3, "same padding output shape");
  // Center (1,1): all 9 neighbors visible → sum = 9
  CHECK_NEAR(out.at(0, 1, 1), 9.0, 1e-9, "same center");
  // Corner (0,0): only 4 elements in window → sum = 4
  CHECK_NEAR(out.at(0, 0, 0), 4.0, 1e-9, "same corner");
  // Edge (0,1): 6 elements visible → sum = 6
  CHECK_NEAR(out.at(0, 0, 1), 6.0, 1e-9, "same edge");
}

//===================================================================================================================//

void testConv2DFullPadding() {
  std::cout << "--- testConv2DFullPadding ---" << std::endl;

  // 1x2x2 input, 1 filter 2x2, stride 1, FULL (pad=1) → 1x3x3 output
  CNN::Tensor3D<double> input({1, 2, 2});
  input.data = {1, 2, 3, 4};

  CNN::ConvLayerConfig config{1, 2, 2, 1, 1, CNN::SlidingStrategyType::FULL};
  CNN::ConvParameters<double> params;
  params.numFilters = 1; params.inputC = 1; params.filterH = 2; params.filterW = 2;
  params.filters.assign(4, 1.0);
  params.biases = {0.0};

  CNN::Tensor3D<double> out = CNN::Conv2D<double>::predict(input, config, params);
  // FULL: pad = 2-1 = 1. outH = (2+2-2)/1+1 = 3
  CHECK(out.shape.c == 1 && out.shape.h == 3 && out.shape.w == 3, "full padding output shape");
  // Corner (0,0): only input[0,0] visible → 1
  CHECK_NEAR(out.at(0, 0, 0), 1.0, 1e-9, "full corner");
  // Center (1,1): all 4 input elements → 1+2+3+4 = 10
  CHECK_NEAR(out.at(0, 1, 1), 10.0, 1e-9, "full center");
  // (0,1): input[0,0]+input[0,1] → 1+2 = 3
  CHECK_NEAR(out.at(0, 0, 1), 3.0, 1e-9, "full top-center");
}

//===================================================================================================================//

void testAvgPool() {
  std::cout << "--- testAvgPool ---" << std::endl;

  CNN::Tensor3D<double> input({1, 4, 4});
  for (ulong i = 0; i < 16; i++) input.data[i] = static_cast<double>(i + 1);

  CNN::PoolLayerConfig config{CNN::PoolTypeEnum::AVG, 2, 2, 2, 2};
  std::vector<ulong> maxIndices;
  CNN::Tensor3D<double> out = CNN::Pool<double>::predict(input, config, maxIndices);

  CHECK(out.shape.c == 1 && out.shape.h == 2 && out.shape.w == 2, "avgpool shape");
  CHECK(maxIndices.empty(), "avgpool no maxIndices");
  // Top-left: (1+2+5+6)/4 = 14/4 = 3.5
  CHECK_NEAR(out.at(0, 0, 0), 3.5, 1e-9, "avgpool top-left");
  // Bottom-right: (11+12+15+16)/4 = 54/4 = 13.5
  CHECK_NEAR(out.at(0, 1, 1), 13.5, 1e-9, "avgpool bottom-right");

  // Backprop: gradient distributed evenly
  CNN::Tensor3D<double> dOut({1, 2, 2}, 4.0);  // gradient = 4.0
  CNN::Tensor3D<double> dIn = CNN::Pool<double>::backpropagate(dOut, input.shape, config, maxIndices);
  // Each element gets 4.0 / (2*2) = 1.0
  CHECK_NEAR(dIn.at(0, 0, 0), 1.0, 1e-9, "avgpool backprop");
  CHECK_NEAR(dIn.at(0, 1, 1), 1.0, 1e-9, "avgpool backprop center");
}

//===================================================================================================================//

void testPoolNonSquare() {
  std::cout << "--- testPoolNonSquare ---" << std::endl;

  // 1x4x6 input, pool 2x3 stride 2x3 → 1x2x2
  CNN::Tensor3D<double> input({1, 4, 6});
  for (ulong i = 0; i < 24; i++) input.data[i] = static_cast<double>(i + 1);
  // Row 0: 1  2  3  4  5  6
  // Row 1: 7  8  9  10 11 12
  // Row 2: 13 14 15 16 17 18
  // Row 3: 19 20 21 22 23 24

  CNN::PoolLayerConfig config{CNN::PoolTypeEnum::MAX, 2, 3, 2, 3};
  std::vector<ulong> maxIndices;
  CNN::Tensor3D<double> out = CNN::Pool<double>::predict(input, config, maxIndices);

  CHECK(out.shape.c == 1 && out.shape.h == 2 && out.shape.w == 2, "non-square pool shape");
  // (oh=0,ow=0): rows 0-1, cols 0-2 → max(1,2,3,7,8,9) = 9
  CHECK_NEAR(out.at(0, 0, 0), 9.0, 1e-9, "non-square pool [0,0]");
  // (oh=0,ow=1): rows 0-1, cols 3-5 → max(4,5,6,10,11,12) = 12
  CHECK_NEAR(out.at(0, 0, 1), 12.0, 1e-9, "non-square pool [0,1]");
  // (oh=1,ow=1): rows 2-3, cols 3-5 → max(16,17,18,22,23,24) = 24
  CHECK_NEAR(out.at(0, 1, 1), 24.0, 1e-9, "non-square pool [1,1]");
}

//===================================================================================================================//

void testSlidingStrategy() {
  std::cout << "--- testSlidingStrategy ---" << std::endl;

  CHECK(CNN::SlidingStrategy::computePadding(3, CNN::SlidingStrategyType::VALID) == 0, "valid pad=0");
  CHECK(CNN::SlidingStrategy::computePadding(5, CNN::SlidingStrategyType::VALID) == 0, "valid pad=0 k5");
  CHECK(CNN::SlidingStrategy::computePadding(3, CNN::SlidingStrategyType::FULL) == 2, "full pad k3");
  CHECK(CNN::SlidingStrategy::computePadding(5, CNN::SlidingStrategyType::FULL) == 4, "full pad k5");
  CHECK(CNN::SlidingStrategy::computePadding(3, CNN::SlidingStrategyType::SAME) == 1, "same pad k3");
  CHECK(CNN::SlidingStrategy::computePadding(5, CNN::SlidingStrategyType::SAME) == 2, "same pad k5");

  CHECK(CNN::SlidingStrategy::nameToType("valid") == CNN::SlidingStrategyType::VALID, "nameToType valid");
  CHECK(CNN::SlidingStrategy::nameToType("same") == CNN::SlidingStrategyType::SAME, "nameToType same");
  CHECK(CNN::SlidingStrategy::nameToType("full") == CNN::SlidingStrategyType::FULL, "nameToType full");

  CHECK(CNN::SlidingStrategy::typeToName(CNN::SlidingStrategyType::VALID) == "valid", "typeToName valid");

  bool caught = false;
  try { CNN::SlidingStrategy::nameToType("bogus"); } catch (const std::runtime_error&) { caught = true; }
  CHECK(caught, "nameToType unknown throws");
}

//===================================================================================================================//

void testValidateShapes() {
  std::cout << "--- testValidateShapes ---" << std::endl;

  CNN::LayersConfig lc;

  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig pool1;
  pool1.type = CNN::LayerType::POOL;
  pool1.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};

  CNN::CNNLayerConfig flatten;
  flatten.type = CNN::LayerType::FLATTEN;
  flatten.config = CNN::FlattenLayerConfig{};

  lc.cnnLayers = {conv1, relu1, pool1, flatten};

  // 1x8x8 → Conv(4,3x3,valid) → 4x6x6 → ReLU → 4x6x6 → Pool(2x2,s2) → 4x3x3
  CNN::Shape3D outShape = lc.validateShapes({1, 8, 8});
  CHECK(outShape.c == 4 && outShape.h == 3 && outShape.w == 3, "validateShapes output");

  // Test with invalid config (filter bigger than input)
  CNN::LayersConfig lc2;
  CNN::CNNLayerConfig badConv;
  badConv.type = CNN::LayerType::CONV;
  badConv.config = CNN::ConvLayerConfig{1, 10, 10, 1, 1, CNN::SlidingStrategyType::VALID};
  lc2.cnnLayers = {badConv};

  bool caught = false;
  try { lc2.validateShapes({1, 3, 3}); } catch (const std::runtime_error&) { caught = true; }
  CHECK(caught, "validateShapes throws for oversized filter");
}

//===================================================================================================================//

void testEndToEnd() {
  std::cout << "--- testEndToEnd (train + predict) ---" << std::endl;

  // 1x5x5 → Conv(1 filter 3x3 valid) → 1x3x3 → ReLU → Flatten(9) → Dense(1, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.verbose = false;

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
  // With all-ones input, conv output = sum(filters) + bias. If sum <= 0, ReLU zeros
  // everything out and gradients are blocked. Using positive filter values ensures
  // the network starts in a learning-friendly state.
  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1);  // All positive → sum = 0.9 > 0
  initConv.biases.assign(1, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.progressReports = 0;

  // "bright" (gradient-fill) → 1, "dark" (all 0s) → 0
  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  // Retry up to 3 times to handle random ANN weight initialization
  std::unique_ptr<CNN::Core<double>> core;
  CNN::Output<double> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    core = CNN::Core<double>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  CHECK(core != nullptr, "core creation");
  CHECK(pred0.size() == 1, "predict output size 0");
  CHECK(pred1.size() == 1, "predict output size 1");
  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "bright > dark after training (5 attempts)");

  // Test method
  CNN::TestResult<double> result = core->test(samples);
  CHECK(result.numSamples == 2, "test numSamples");
  CHECK(result.averageLoss >= 0.0, "test avgLoss non-negative");
  std::cout << "  test avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

void testGPUEndToEnd() {
  std::cout << "--- testGPUEndToEnd (train + predict) ---" << std::endl;

  // Same architecture as CPU test:
  // 1x5x5 → Conv(1 filter 3x3 valid) → 1x3x3 → ReLU → Flatten(9) → Dense(1, sigmoid)
  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 5, 5};
  config.verbose = false;

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

  // Pre-initialized conv parameters (same as CPU test)
  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.progressReports = 0;
  config.trainingConfig.numGPUs = 1;

  // "bright" (gradient-fill) → 1, "dark" (all 0s) → 0
  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  // Retry up to 3 times to handle random ANN weight initialization
  std::unique_ptr<CNN::Core<float>> core;
  CNN::Output<float> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    core = CNN::Core<float>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > 0.7f && pred1[0] < 0.3f) converged = true;
  }

  CHECK(core != nullptr, "GPU core creation");
  CHECK(pred0.size() == 1, "GPU predict output size 0");
  CHECK(pred1.size() == 1, "GPU predict output size 1");
  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "GPU bright > 0.7 & dark < 0.3 after training (5 attempts)");

  // Test method
  CNN::TestResult<float> result = core->test(samples);
  CHECK(result.numSamples == 2, "GPU test numSamples");
  CHECK(result.averageLoss >= 0.0f, "GPU test avgLoss non-negative");
  CHECK(result.averageLoss < 0.1f, "GPU test avgLoss reasonably small");
  std::cout << "  test avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

void testGPUPredictOnly() {
  std::cout << "--- testGPUPredictOnly ---" << std::endl;

  // Train on CPU, then verify GPU predict gives same results
  CNN::CoreConfig<float> cpuConfig;
  cpuConfig.modeType = CNN::ModeType::TRAIN;
  cpuConfig.deviceType = CNN::DeviceType::CPU;
  cpuConfig.inputShape = {1, 5, 5};
  cpuConfig.verbose = false;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  cpuConfig.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  cpuConfig.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1f);
  initConv.biases.assign(1, 0.0f);
  cpuConfig.parameters.convParams = {initConv};

  cpuConfig.trainingConfig.numEpochs = 50;
  cpuConfig.trainingConfig.learningRate = 0.5f;
  cpuConfig.trainingConfig.progressReports = 0;

  auto cpuCore = CNN::Core<float>::makeCore(cpuConfig);

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 5, 5});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 5, 5}, 0.0f);
  samples[1].output = {0.0f};

  cpuCore->train(samples);

  // Get trained parameters from CPU core
  const CNN::Parameters<float>& trainedParams = cpuCore->getParameters();

  // Create a GPU core in PREDICT mode with the trained parameters
  CNN::CoreConfig<float> gpuConfig;
  gpuConfig.modeType = CNN::ModeType::PREDICT;
  gpuConfig.deviceType = CNN::DeviceType::GPU;
  gpuConfig.inputShape = {1, 5, 5};
  gpuConfig.verbose = false;
  gpuConfig.layersConfig = cpuConfig.layersConfig;
  gpuConfig.parameters = trainedParams;

  auto gpuCore = CNN::Core<float>::makeCore(gpuConfig);

  CNN::Output<float> cpuPred0 = cpuCore->predict(samples[0].input);
  CNN::Output<float> gpuPred0 = gpuCore->predict(samples[0].input);
  CNN::Output<float> cpuPred1 = cpuCore->predict(samples[1].input);
  CNN::Output<float> gpuPred1 = gpuCore->predict(samples[1].input);

  std::cout << "  CPU bright=" << cpuPred0[0] << "  GPU bright=" << gpuPred0[0] << std::endl;
  std::cout << "  CPU dark=" << cpuPred1[0] << "  GPU dark=" << gpuPred1[0] << std::endl;

  // GPU and CPU should produce very close results (float precision)
  CHECK_NEAR(cpuPred0[0], gpuPred0[0], 0.01f, "GPU vs CPU bright prediction");
  CHECK_NEAR(cpuPred1[0], gpuPred1[0], 0.01f, "GPU vs CPU dark prediction");
}

//===================================================================================================================//

void testMultiConvStack() {
  std::cout << "--- testMultiConvStack (Conv→ReLU→Conv→ReLU→Flatten→Dense) ---" << std::endl;

  // 1x8x8 → Conv(2,3x3) → 2x6x6 → ReLU → Conv(1,3x3) → 1x4x4 → ReLU → Flatten(16) → Dense(1,sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 8, 8};
  config.verbose = false;

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
  initConv1.numFilters = 2; initConv1.inputC = 1; initConv1.filterH = 3; initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1);
  initConv1.biases.assign(2, 0.0);

  CNN::ConvParameters<double> initConv2;
  initConv2.numFilters = 1; initConv2.inputC = 2; initConv2.filterH = 3; initConv2.filterW = 3;
  initConv2.filters.assign(1 * 2 * 3 * 3, 0.1);
  initConv2.biases.assign(1, 0.0);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 8, 8});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 8, 8}, 0.0);
  samples[1].output = {0.0};

  // Retry up to 3 times to handle random ANN weight initialization
  CNN::Output<double> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "multi-conv bright > dark (5 attempts)");
}

//===================================================================================================================//

void testConvPoolConv() {
  std::cout << "--- testConvPoolConv (Conv→ReLU→Pool→Conv→ReLU→Flatten→Dense) ---" << std::endl;

  // 1x10x10 → Conv(2,3x3) → 2x8x8 → ReLU → MaxPool(2x2,s2) → 2x4x4 → Conv(1,3x3) → 1x2x2 → ReLU → Flatten(4) → Dense(1,sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 10, 10};
  config.verbose = false;

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
  initConv1.numFilters = 2; initConv1.inputC = 1; initConv1.filterH = 3; initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1);
  initConv1.biases.assign(2, 0.0);

  CNN::ConvParameters<double> initConv2;
  initConv2.numFilters = 1; initConv2.inputC = 2; initConv2.filterH = 3; initConv2.filterW = 3;
  initConv2.filters.assign(1 * 2 * 3 * 3, 0.1);
  initConv2.biases.assign(1, 0.0);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 10, 10});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 10, 10}, 0.0);
  samples[1].output = {0.0};

  // Retry up to 3 times to handle random ANN weight initialization
  CNN::Output<double> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "conv-pool-conv bright > dark (5 attempts)");
}

//===================================================================================================================//

void testMultiChannelInput() {
  std::cout << "--- testMultiChannelInput (3-channel RGB-like) ---" << std::endl;

  // 3x6x6 → Conv(2,3x3) → 2x4x4 → ReLU → Flatten(32) → Dense(1,sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {3, 6, 6};
  config.verbose = false;

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
  initConv.numFilters = 2; initConv.inputC = 3; initConv.filterH = 3; initConv.filterW = 3;
  initConv.filters.assign(2 * 3 * 3 * 3, 0.05);  // small positive values
  initConv.biases.assign(2, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.trainingConfig.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({3, 6, 6}, 0.3, 1.0);
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({3, 6, 6}, 0.0);
  samples[1].output = {0.0};

  // Retry up to 3 times to handle random ANN weight initialization
  CNN::Output<double> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "multichannel bright > dark (5 attempts)");
}

//===================================================================================================================//

void testParameterRoundTrip() {
  std::cout << "--- testParameterRoundTrip ---" << std::endl;

  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.verbose = false;

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
  initConv.numFilters = 1; initConv.inputC = 1; initConv.filterH = 3; initConv.filterW = 3;
  initConv.filters.assign(9, 0.1);
  initConv.biases.assign(1, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.progressReports = 0;

  auto core = CNN::Core<double>::makeCore(config);

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  core->train(samples);

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
  predictConfig.verbose = false;
  predictConfig.layersConfig = config.layersConfig;
  predictConfig.parameters = params;

  auto predictCore = CNN::Core<double>::makeCore(predictConfig);
  CHECK(predictCore != nullptr, "param roundtrip predict core creation");

  // Predictions should match
  CNN::Output<double> origPred = core->predict(samples[0].input);
  CNN::Output<double> newPred = predictCore->predict(samples[0].input);
  CHECK_NEAR(origPred[0], newPred[0], 1e-9, "param roundtrip prediction match");
  std::cout << "  original=" << origPred[0] << "  from_params=" << newPred[0] << std::endl;
}

//===================================================================================================================//

void testMultipleOutputNeurons() {
  std::cout << "--- testMultipleOutputNeurons ---" << std::endl;

  // 1x8x8 → Conv(2,3x3) → ReLU → Flatten(72) → Dense(3, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::CPU;
  config.inputShape = {1, 8, 8};
  config.verbose = false;

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
  initConv.numFilters = 2; initConv.inputC = 1; initConv.filterH = 3; initConv.filterW = 3;
  initConv.filters.assign(2 * 9, 0.1);
  initConv.biases.assign(2, 0.0);
  config.parameters.convParams = {initConv};

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.progressReports = 0;

  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 8, 8});
  samples[0].output = {1.0, 0.0, 1.0};  // target: [1, 0, 1]
  samples[1].input = CNN::Tensor3D<double>({1, 8, 8}, 0.0);
  samples[1].output = {0.0, 1.0, 0.0};  // target: [0, 1, 0]

  // Retry up to 3 times to handle random ANN weight initialization
  CNN::Output<double> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<double>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  CHECK(pred0.size() == 3, "multi-output size");
  std::cout << "  pred(bright)=[" << pred0[0] << "," << pred0[1] << "," << pred0[2] << "]" << std::endl;
  CHECK(converged, "multi-output[0] bright > dark (5 attempts)");
}

//===================================================================================================================//

void testGPUWithPoolLayer() {
  std::cout << "--- testGPUWithPoolLayer (Conv→ReLU→Pool→Conv→ReLU→Flatten→Dense) ---" << std::endl;

  // Same architecture as testConvPoolConv but on GPU
  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 10, 10};
  config.verbose = false;

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

  CNN::ConvParameters<float> initConv1;
  initConv1.numFilters = 2; initConv1.inputC = 1; initConv1.filterH = 3; initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1f);
  initConv1.biases.assign(2, 0.0f);

  CNN::ConvParameters<float> initConv2;
  initConv2.numFilters = 1; initConv2.inputC = 2; initConv2.filterH = 3; initConv2.filterW = 3;
  initConv2.filters.assign(1 * 2 * 3 * 3, 0.1f);
  initConv2.biases.assign(1, 0.0f);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.trainingConfig.progressReports = 0;
  config.trainingConfig.numGPUs = 1;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 10, 10});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 10, 10}, 0.0f);
  samples[1].output = {0.0f};

  // Retry up to 3 times to handle random ANN weight initialization
  CNN::Output<float> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "GPU pool bright > dark (5 attempts)");
}

//===================================================================================================================//

void testGPUMultiConvStack() {
  std::cout << "--- testGPUMultiConvStack (Conv→ReLU→Conv→ReLU→Flatten→Dense) ---" << std::endl;

  CNN::CoreConfig<float> config;
  config.modeType = CNN::ModeType::TRAIN;
  config.deviceType = CNN::DeviceType::GPU;
  config.inputShape = {1, 8, 8};
  config.verbose = false;

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

  CNN::ConvParameters<float> initConv1;
  initConv1.numFilters = 2; initConv1.inputC = 1; initConv1.filterH = 3; initConv1.filterW = 3;
  initConv1.filters.assign(2 * 1 * 3 * 3, 0.1f);
  initConv1.biases.assign(2, 0.0f);

  CNN::ConvParameters<float> initConv2;
  initConv2.numFilters = 1; initConv2.inputC = 2; initConv2.filterH = 3; initConv2.filterW = 3;
  initConv2.filters.assign(1 * 2 * 3 * 3, 0.1f);
  initConv2.biases.assign(1, 0.0f);

  config.parameters.convParams = {initConv1, initConv2};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.trainingConfig.progressReports = 0;
  config.trainingConfig.numGPUs = 1;

  CNN::Samples<float> samples(2);
  samples[0].input = makeGradientInput<float>({1, 8, 8});
  samples[0].output = {1.0f};
  samples[1].input = CNN::Tensor3D<float>({1, 8, 8}, 0.0f);
  samples[1].output = {0.0f};

  // Retry up to 3 times to handle random ANN weight initialization
  CNN::Output<float> pred0, pred1;
  bool converged = false;
  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = CNN::Core<float>::makeCore(config);
    core->train(samples);
    pred0 = core->predict(samples[0].input);
    pred1 = core->predict(samples[1].input);
    if (pred0[0] > pred1[0]) converged = true;
  }

  std::cout << "  pred(bright)=" << pred0[0] << "  pred(dark)=" << pred1[0] << std::endl;
  CHECK(converged, "GPU multi-conv bright > dark (5 attempts)");
}

//===================================================================================================================//

int main() {
  std::cout << "=== CNN Unit Tests ===" << std::endl;

  testTensor3D();
  testConv2DForward();
  testConv2DBackprop();
  testConv2DWithBias();
  testConv2DMultiFilter();
  testConv2DMultiChannel();
  testConv2DBackpropValues();
  testConv2DStride();
  testConv2DSamePadding();
  testConv2DFullPadding();
  testReLU();
  testMaxPool();
  testAvgPool();
  testPoolNonSquare();
  testFlatten();
  testSlidingStrategy();
  testValidateShapes();

  std::cout << std::endl;
  std::cout << "=== Integration Tests ===" << std::endl;
  testEndToEnd();
  testMultiConvStack();
  testConvPoolConv();
  testMultiChannelInput();
  testParameterRoundTrip();
  testMultipleOutputNeurons();

  std::cout << std::endl;
  std::cout << "=== GPU Tests ===" << std::endl;
  testGPUEndToEnd();
  testGPUPredictOnly();
  testGPUWithPoolLayer();
  testGPUMultiConvStack();

  std::cout << std::endl;
  std::cout << "=== Results: " << testsPassed << " passed, " << testsFailed << " failed ===" << std::endl;

  return (testsFailed > 0) ? 1 : 0;
}