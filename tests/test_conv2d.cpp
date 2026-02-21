#include "test_helpers.hpp"

//===================================================================================================================//

static void testConv2DForward() {
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

static void testConv2DBackprop() {
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

static void testConv2DWithBias() {
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

static void testConv2DMultiFilter() {
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

static void testConv2DMultiChannel() {
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

static void testConv2DBackpropValues() {
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

static void testConv2DStride() {
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

static void testConv2DSamePadding() {
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

static void testConv2DFullPadding() {
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

void runConv2DTests() {
  testConv2DForward();
  testConv2DBackprop();
  testConv2DWithBias();
  testConv2DMultiFilter();
  testConv2DMultiChannel();
  testConv2DBackpropValues();
  testConv2DStride();
  testConv2DSamePadding();
  testConv2DFullPadding();
}

