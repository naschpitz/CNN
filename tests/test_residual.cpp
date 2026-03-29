#include "test_helpers.hpp"

#include "CNN_LayersConfig.hpp"
#include "CNN_Residual.hpp"

//===================================================================================================================//

static void testResidualIdentityForward()
{
  std::cout << "--- testResidualIdentityForward ---" << std::endl;

  CNN::Shape3D shape{2, 3, 3};
  CNN::Tensor3D<double> blockOutput(shape);
  CNN::Tensor3D<double> skipInput(shape);

  for (ulong i = 0; i < 18; i++) {
    blockOutput.data[i] = static_cast<double>(i) * 0.1;
    skipInput.data[i] = static_cast<double>(i) * 0.2;
  }

  CNN::Residual<double>::propagate(blockOutput, skipInput, nullptr);

  for (ulong i = 0; i < 18; i++) {
    double expected = static_cast<double>(i) * 0.1 + static_cast<double>(i) * 0.2;
    CHECK_NEAR(blockOutput.data[i], expected, 1e-9, "res identity forward");
  }
}

//===================================================================================================================//

static void testResidualIdentityBackward()
{
  std::cout << "--- testResidualIdentityBackward ---" << std::endl;

  CNN::Shape3D shape{2, 3, 3};
  CNN::Tensor3D<double> dOut(shape);
  CNN::Tensor3D<double> skipInput(shape);

  for (ulong i = 0; i < 18; i++) {
    dOut.data[i] = static_cast<double>(i) * 0.5;
    skipInput.data[i] = 0.0; // Not used for identity
  }

  CNN::Tensor3D<double> dSkip = CNN::Residual<double>::backpropagate(dOut, skipInput, nullptr, nullptr);

  // Identity: dSkip == dOut
  for (ulong i = 0; i < 18; i++)
    CHECK_NEAR(dSkip.data[i], dOut.data[i], 1e-9, "res identity backward");
}

//===================================================================================================================//

static void testResidualParametersForward()
{
  std::cout << "--- testResidualParametersForward ---" << std::endl;

  // Skip: 2ch, Block output: 4ch, spatial 2x2
  CNN::Shape3D skipShape{2, 2, 2};
  CNN::Shape3D outShape{4, 2, 2};
  CNN::Tensor3D<double> blockOutput(outShape, 0.0);
  CNN::Tensor3D<double> skipInput(skipShape);

  for (ulong i = 0; i < 8; i++)
    skipInput.data[i] = static_cast<double>(i + 1);

  CNN::ResidualParameters<double> proj;
  proj.inC = 2;
  proj.outC = 4;
  proj.weights.resize(8, 0.1); // 4x2 all 0.1
  proj.biases.resize(4, 0.0);

  CNN::Residual<double>::propagate(blockOutput, skipInput, &proj);

  for (ulong oc = 0; oc < 4; oc++) {
    for (ulong s = 0; s < 4; s++) {
      double ic0val = skipInput.data[s];
      double ic1val = skipInput.data[4 + s];
      double expected = 0.1 * ic0val + 0.1 * ic1val;
      CHECK_NEAR(blockOutput.data[oc * 4 + s], expected, 1e-9, "res projection forward");
    }
  }
}

//===================================================================================================================//

static void testResidualParametersBackward()
{
  std::cout << "--- testResidualParametersBackward ---" << std::endl;

  CNN::Shape3D skipShape{2, 2, 2};
  CNN::Shape3D outShape{4, 2, 2};
  CNN::Tensor3D<double> dOut(outShape);
  CNN::Tensor3D<double> skipInput(skipShape);

  for (ulong i = 0; i < 16; i++)
    dOut.data[i] = static_cast<double>(i + 1) * 0.1;

  for (ulong i = 0; i < 8; i++)
    skipInput.data[i] = static_cast<double>(i + 1);

  CNN::ResidualParameters<double> proj;
  proj.inC = 2;
  proj.outC = 4;
  proj.weights.resize(8, 0.1);
  proj.biases.resize(4, 0.0);

  CNN::ResidualParameters<double> dProj;
  dProj.inC = 2;
  dProj.outC = 4;
  dProj.weights.assign(8, 0.0);
  dProj.biases.assign(4, 0.0);

  CNN::Tensor3D<double> dSkip = CNN::Residual<double>::backpropagate(dOut, skipInput, &proj, &dProj);

  for (ulong ic = 0; ic < 2; ic++) {
    for (ulong s = 0; s < 4; s++) {
      double sumOc = 0.0;

      for (ulong oc = 0; oc < 4; oc++)
        sumOc += dOut.data[oc * 4 + s];

      double expected = 0.1 * sumOc;
      CHECK_NEAR(dSkip.data[ic * 4 + s], expected, 1e-9, "res proj backward dSkip");
    }
  }

  for (ulong oc = 0; oc < 4; oc++) {
    for (ulong ic = 0; ic < 2; ic++) {
      double expected = 0.0;

      for (ulong s = 0; s < 4; s++)
        expected += dOut.data[oc * 4 + s] * skipInput.data[ic * 4 + s];

      CHECK_NEAR(dProj.weights[oc * 2 + ic], expected, 1e-9, "res proj backward dW");
    }
  }
}



//===================================================================================================================//

static void testResidualGradientCheck()
{
  std::cout << "--- testResidualGradientCheck ---" << std::endl;

  CNN::Shape3D skipShape{2, 2, 2};
  CNN::Shape3D outShape{3, 2, 2};
  ulong skipSize = 8;
  ulong outSize = 12;
  double eps = 1e-5;

  std::vector<double> skipData(skipSize), blockData(outSize);

  for (ulong i = 0; i < skipSize; i++)
    skipData[i] = static_cast<double>(i) * 0.1 - 0.3;

  for (ulong i = 0; i < outSize; i++)
    blockData[i] = static_cast<double>(i) * 0.05 + 0.1;

  CNN::ResidualParameters<double> proj;
  proj.inC = 2;
  proj.outC = 3;
  proj.weights = {0.1, -0.2, 0.3, -0.1, 0.2, -0.3};
  proj.biases = {0.01, -0.02, 0.03};

  std::vector<double> dOut(outSize);

  for (ulong i = 0; i < outSize; i++)
    dOut[i] = static_cast<double>(i) * 0.2 - 0.5;

  CNN::Tensor3D<double> dOutTensor(outShape);
  dOutTensor.data = dOut;
  CNN::Tensor3D<double> skipInputTensor(skipShape);
  skipInputTensor.data = skipData;

  CNN::ResidualParameters<double> dProj;
  dProj.inC = 2;
  dProj.outC = 3;
  dProj.weights.assign(6, 0.0);
  dProj.biases.assign(3, 0.0);

  CNN::Tensor3D<double> dSkip = CNN::Residual<double>::backpropagate(dOutTensor, skipInputTensor, &proj, &dProj);

  for (ulong i = 0; i < skipSize; i++) {
    auto fwd = [&](std::vector<double>& sd) {
      CNN::Tensor3D<double> bo(outShape);
      bo.data = blockData;
      CNN::Tensor3D<double> si(skipShape);
      si.data = sd;
      CNN::Residual<double>::propagate(bo, si, &proj);

      double loss = 0.0;

      for (ulong j = 0; j < outSize; j++)
        loss += dOut[j] * bo.data[j];

      return loss;
    };

    auto plus = skipData;
    plus[i] += eps;
    auto minus = skipData;
    minus[i] -= eps;

    double numGrad = (fwd(plus) - fwd(minus)) / (2.0 * eps);
    CHECK_NEAR(dSkip.data[i], numGrad, 1e-6, "res proj numerical grad skip");
  }
}

//===================================================================================================================//

static void testResidualShapeValidation()
{
  std::cout << "--- testResidualShapeValidation ---" << std::endl;

  // 1. Unmatched residual_end (no start)
  {
    CNN::LayersConfig lc;
    CNN::CNNLayerConfig resEnd;
    resEnd.type = CNN::LayerType::RESIDUAL_END;
    resEnd.config = CNN::ResidualEndConfig{};
    lc.cnnLayers = {resEnd};

    bool threw = false;

    try {
      lc.validateShapes({4, 8, 8});
    } catch (const std::runtime_error& e) {
      threw = true;
      std::string msg = e.what();
      CHECK(msg.find("no matching residual_start") != std::string::npos, "unmatched end error message");
    }

    CHECK(threw, "unmatched residual_end should throw");
  }

  // 2. Unmatched residual_start (no end)
  {
    CNN::LayersConfig lc;
    CNN::CNNLayerConfig resStart;
    resStart.type = CNN::LayerType::RESIDUAL_START;
    resStart.config = CNN::ResidualStartConfig{};
    lc.cnnLayers = {resStart};

    bool threw = false;

    try {
      lc.validateShapes({4, 8, 8});
    } catch (const std::runtime_error& e) {
      threw = true;
      std::string msg = e.what();
      CHECK(msg.find("Unmatched residual_start") != std::string::npos, "unmatched start error message");
    }

    CHECK(threw, "unmatched residual_start should throw");
  }

  // 3. Spatial dimension mismatch (pool inside residual block)
  {
    CNN::LayersConfig lc;
    CNN::CNNLayerConfig resStart;
    resStart.type = CNN::LayerType::RESIDUAL_START;
    resStart.config = CNN::ResidualStartConfig{};

    CNN::CNNLayerConfig pool;
    pool.type = CNN::LayerType::POOL;
    pool.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};

    CNN::CNNLayerConfig resEnd;
    resEnd.type = CNN::LayerType::RESIDUAL_END;
    resEnd.config = CNN::ResidualEndConfig{};

    lc.cnnLayers = {resStart, pool, resEnd};

    bool threw = false;

    try {
      lc.validateShapes({4, 8, 8});
    } catch (const std::runtime_error& e) {
      threw = true;
      std::string msg = e.what();
      CHECK(msg.find("spatial dimension mismatch") != std::string::npos, "spatial mismatch error message");
    }

    CHECK(threw, "spatial mismatch should throw");
  }

  // 4. Empty residual block (start immediately followed by end) — should NOT throw
  {
    CNN::LayersConfig lc;
    CNN::CNNLayerConfig resStart;
    resStart.type = CNN::LayerType::RESIDUAL_START;
    resStart.config = CNN::ResidualStartConfig{};

    CNN::CNNLayerConfig resEnd;
    resEnd.type = CNN::LayerType::RESIDUAL_END;
    resEnd.config = CNN::ResidualEndConfig{};

    lc.cnnLayers = {resStart, resEnd};

    bool threw = false;

    try {
      lc.validateShapes({4, 8, 8});
    } catch (...) {
      threw = true;
    }

    CHECK(!threw, "empty residual block should not throw");
  }

  // 5. Channel mismatch with same spatial — should NOT throw (projection auto-created)
  {
    CNN::LayersConfig lc;
    CNN::CNNLayerConfig resStart;
    resStart.type = CNN::LayerType::RESIDUAL_START;
    resStart.config = CNN::ResidualStartConfig{};

    CNN::CNNLayerConfig conv;
    conv.type = CNN::LayerType::CONV;
    conv.config = CNN::ConvLayerConfig{8, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

    CNN::CNNLayerConfig resEnd;
    resEnd.type = CNN::LayerType::RESIDUAL_END;
    resEnd.config = CNN::ResidualEndConfig{};

    lc.cnnLayers = {resStart, conv, resEnd};

    bool threw = false;

    try {
      lc.validateShapes({4, 8, 8});
    } catch (...) {
      threw = true;
    }

    CHECK(!threw, "channel mismatch with same spatial should not throw");
  }
}

//===================================================================================================================//

void runResidualTests()
{
  testResidualIdentityForward();
  testResidualIdentityBackward();
  testResidualParametersForward();
  testResidualParametersBackward();
  testResidualGradientCheck();
  testResidualShapeValidation();
}