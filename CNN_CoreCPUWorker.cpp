#include "CNN_CoreCPUWorker.hpp"
#include "CNN_Conv2D.hpp"
#include "CNN_ReLU.hpp"
#include "CNN_Pool.hpp"
#include "CNN_Flatten.hpp"
#include "CNN_GlobalAvgPool.hpp"
#include "CNN_GlobalDualPool.hpp"
#include "CNN_Normalization.hpp"
#include "CNN_Residual.hpp"

#include <ANN_Core.hpp>
#include <stack>

using namespace CNN;

//===================================================================================================================//

template <typename T>
CoreCPUWorker<T>::CoreCPUWorker(const CoreConfig<T>& config, const LayersConfig& layersConfig,
                                const Parameters<T>& sharedParams, bool allocateTraining)
  : layersConfig(layersConfig),
    sharedParams(sharedParams)
{
  this->costFunctionConfig = config.costFunctionConfig;

  // Compute CNN output shape
  this->cnnOutputShape = this->layersConfig.validateShapes(config.inputShape);
  this->flattenSize = this->cnnOutputShape.size();

  // Build and create ANN sub-core
  ANN::CoreConfig<T> annConfig = buildANNConfig(config, this->flattenSize);
  this->annCore = ANN::Core<T>::makeCore(annConfig);

  // Allocate CNN gradient accumulators if training
  if (allocateTraining) {
    this->accumDConvFilters.resize(sharedParams.convParams.size());
    this->accumDConvBiases.resize(sharedParams.convParams.size());

    for (ulong i = 0; i < sharedParams.convParams.size(); i++) {
      this->accumDConvFilters[i].resize(sharedParams.convParams[i].filters.size(), static_cast<T>(0));
      this->accumDConvBiases[i].resize(sharedParams.convParams[i].biases.size(), static_cast<T>(0));
    }

    this->accumDBNGamma.resize(sharedParams.normParams.size());
    this->accumDBNBeta.resize(sharedParams.normParams.size());
    this->accumNormMean.resize(sharedParams.normParams.size());
    this->accumNormVar.resize(sharedParams.normParams.size());

    for (ulong i = 0; i < sharedParams.normParams.size(); i++) {
      this->accumDBNGamma[i].resize(sharedParams.normParams[i].numChannels, static_cast<T>(0));
      this->accumDBNBeta[i].resize(sharedParams.normParams[i].numChannels, static_cast<T>(0));
      this->accumNormMean[i].resize(sharedParams.normParams[i].numChannels, static_cast<T>(0));
      this->accumNormVar[i].resize(sharedParams.normParams[i].numChannels, static_cast<T>(0));
    }

    this->accumDResidualWeights.resize(sharedParams.residualParams.size());
    this->accumDResidualBiases.resize(sharedParams.residualParams.size());

    for (ulong i = 0; i < sharedParams.residualParams.size(); i++) {
      this->accumDResidualWeights[i].resize(sharedParams.residualParams[i].weights.size(), static_cast<T>(0));
      this->accumDResidualBiases[i].resize(sharedParams.residualParams[i].biases.size(), static_cast<T>(0));
    }

    this->bnSampleCount = 0;
  }
}

//===================================================================================================================//

template <typename T>
ANN::CoreConfig<T> CoreCPUWorker<T>::buildANNConfig(const CoreConfig<T>& cnnConfig, ulong flattenSize)
{
  ANN::CoreConfig<T> annConfig;

  // Map CNN mode to ANN mode
  switch (cnnConfig.modeType) {
  case ModeType::TRAIN:
    annConfig.modeType = ANN::ModeType::TRAIN;
    break;
  case ModeType::TEST:
    annConfig.modeType = ANN::ModeType::TEST;
    break;
  default:
    annConfig.modeType = ANN::ModeType::PREDICT;
    break;
  }

  annConfig.deviceType = ANN::DeviceType::CPU;

  // Build ANN layers config: first layer = flatten size (input), rest from denseLayersConfig
  ANN::LayersConfig annLayers;

  ANN::Layer inputLayer;
  inputLayer.numNeurons = flattenSize;
  inputLayer.actvFuncType = ANN::ActvFuncType::RELU; // Placeholder (ANN input layer activation unused)
  annLayers.push_back(inputLayer);

  for (const auto& denseConfig : cnnConfig.layersConfig.denseLayers) {
    ANN::Layer layer;
    layer.numNeurons = denseConfig.numNeurons;
    layer.actvFuncType = denseConfig.actvFuncType;
    annLayers.push_back(layer);
  }

  annConfig.layersConfig = annLayers;

  annConfig.trainingConfig.numEpochs = cnnConfig.trainingConfig.numEpochs;
  annConfig.trainingConfig.learningRate = cnnConfig.trainingConfig.learningRate;
  annConfig.trainingConfig.dropoutRate = cnnConfig.trainingConfig.dropoutRate;
  annConfig.trainingConfig.optimizer.type = static_cast<ANN::OptimizerType>(cnnConfig.trainingConfig.optimizer.type);
  annConfig.trainingConfig.optimizer.beta1 = cnnConfig.trainingConfig.optimizer.beta1;
  annConfig.trainingConfig.optimizer.beta2 = cnnConfig.trainingConfig.optimizer.beta2;
  annConfig.trainingConfig.optimizer.epsilon = cnnConfig.trainingConfig.optimizer.epsilon;
  annConfig.numThreads = 1; // CNN manages its own threading

  annConfig.costFunctionConfig.type = static_cast<ANN::CostFunctionType>(cnnConfig.costFunctionConfig.type);
  annConfig.costFunctionConfig.weights = cnnConfig.costFunctionConfig.weights;

  annConfig.parameters = cnnConfig.parameters.denseParams;

  annConfig.logLevel = static_cast<ANN::LogLevel>(cnnConfig.logLevel);
  annConfig.progressReports = 0;

  return annConfig;
}

//===================================================================================================================//

template <typename T>
Output<T> CoreCPUWorker<T>::predict(const Input<T>& input)
{
  Tensor3D<T> cnnOut = this->propagateCNN(input);
  Tensor1D<T> flatInput = Flatten<T>::propagate(cnnOut);

  ANN::Input<T> annInput(flatInput.begin(), flatInput.end());
  ANN::Output<T> annOutput = this->annCore->predict(annInput);

  return Output<T>(annOutput.begin(), annOutput.end());
}

//===================================================================================================================//

template <typename T>
T CoreCPUWorker<T>::processSample(const Input<T>& input, const Output<T>& expected)
{
  // CNN propagate (with intermediates for backpropagation)
  std::vector<Tensor3D<T>> intermediates;
  std::vector<std::vector<ulong>> poolMaxIndices;
  Tensor3D<T> cnnOut = this->propagateCNN(input, true, &intermediates, &poolMaxIndices);
  Tensor1D<T> flatInput = Flatten<T>::propagate(cnnOut);

  // ANN propagate
  ANN::Input<T> annInput(flatInput.begin(), flatInput.end());
  ANN::Output<T> annOutput = this->annCore->predict(annInput);
  Output<T> predicted(annOutput.begin(), annOutput.end());

  // Loss
  T sampleLoss = this->calculateLoss(predicted, expected);

  // ANN backpropagate + accumulate
  ANN::Output<T> annExpected(expected.begin(), expected.end());
  ANN::Tensor1D<T> dFlatInput = this->annCore->backpropagate(annExpected);
  this->annCore->accumulate();

  // CNN backpropagate
  Tensor1D<T> dFlat(dFlatInput.begin(), dFlatInput.end());
  Tensor3D<T> dCNNOut = Flatten<T>::backpropagate(dFlat, this->cnnOutputShape);
  std::vector<std::vector<T>> dConvFilters, dConvBiases;
  std::vector<std::vector<T>> dBNGamma, dBNBeta;
  this->backpropagateCNN(dCNNOut, intermediates, poolMaxIndices, dConvFilters, dConvBiases, dBNGamma, dBNBeta);

  // Accumulate CNN gradients
  for (ulong i = 0; i < dConvFilters.size(); i++) {
    for (ulong j = 0; j < dConvFilters[i].size(); j++)
      this->accumDConvFilters[i][j] += dConvFilters[i][j];

    for (ulong j = 0; j < dConvBiases[i].size(); j++)
      this->accumDConvBiases[i][j] += dConvBiases[i][j];
  }

  // Accumulate residual projection gradients
  for (ulong i = 0; i < this->dResidualWeights.size(); i++) {
    for (ulong j = 0; j < this->dResidualWeights[i].size(); j++)
      this->accumDResidualWeights[i][j] += this->dResidualWeights[i][j];

    for (ulong j = 0; j < this->dResidualBiases[i].size(); j++)
      this->accumDResidualBiases[i][j] += this->dResidualBiases[i][j];
  }

  // Accumulate batch norm gradients and running stats
  for (ulong i = 0; i < dBNGamma.size(); i++) {
    for (ulong j = 0; j < dBNGamma[i].size(); j++)
      this->accumDBNGamma[i][j] += dBNGamma[i][j];

    for (ulong j = 0; j < dBNBeta[i].size(); j++)
      this->accumDBNBeta[i][j] += dBNBeta[i][j];

    // With N=1 per-sample, statsMean/statsVar are [1*C], indexed same as [C]
    for (ulong j = 0; j < this->normStatsMean[i].size(); j++)
      this->accumNormMean[i][j] += this->normStatsMean[i][j];

    for (ulong j = 0; j < this->normStatsVar[i].size(); j++)
      this->accumNormVar[i][j] += this->normStatsVar[i][j];
  }

  this->bnSampleCount++;
  this->accum_loss += sampleLoss;

  return sampleLoss;
}

//===================================================================================================================//

template <typename T>
void CoreCPUWorker<T>::resetAccumulators()
{
  for (ulong i = 0; i < this->accumDConvFilters.size(); i++) {
    std::fill(this->accumDConvFilters[i].begin(), this->accumDConvFilters[i].end(), static_cast<T>(0));
    std::fill(this->accumDConvBiases[i].begin(), this->accumDConvBiases[i].end(), static_cast<T>(0));
  }

  for (ulong i = 0; i < this->accumDBNGamma.size(); i++) {
    std::fill(this->accumDBNGamma[i].begin(), this->accumDBNGamma[i].end(), static_cast<T>(0));
    std::fill(this->accumDBNBeta[i].begin(), this->accumDBNBeta[i].end(), static_cast<T>(0));
    std::fill(this->accumNormMean[i].begin(), this->accumNormMean[i].end(), static_cast<T>(0));
    std::fill(this->accumNormVar[i].begin(), this->accumNormVar[i].end(), static_cast<T>(0));
  }

  for (ulong i = 0; i < this->accumDResidualWeights.size(); i++) {
    std::fill(this->accumDResidualWeights[i].begin(), this->accumDResidualWeights[i].end(), static_cast<T>(0));
    std::fill(this->accumDResidualBiases[i].begin(), this->accumDResidualBiases[i].end(), static_cast<T>(0));
  }

  this->bnSampleCount = 0;
  this->annCore->resetAccumulators();
}

//===================================================================================================================//

template <typename T>
Tensor3D<T> CoreCPUWorker<T>::propagateCNN(const Input<T>& input, bool training,
                                           std::vector<Tensor3D<T>>* intermediates,
                                           std::vector<std::vector<ulong>>* poolMaxIndices)
{
  if (training) {
    intermediates->clear();
    poolMaxIndices->clear();
    this->normXNormalized.clear();
    this->normStatsMean.clear();
    this->normStatsVar.clear();
  }

  Tensor3D<T> current = input;
  ulong convIdx = 0;
  ulong normIdx = 0;
  ulong residualIdx = 0;
  std::stack<Tensor3D<T>> residualStack;

  for (const auto& layerConfig : this->layersConfig.cnnLayers) {
    if (training)
      intermediates->push_back(current);

    switch (layerConfig.type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      current = Conv2D<T>::propagate(current, conv, this->sharedParams.convParams[convIdx]);
      convIdx++;
      break;
    }

    case LayerType::RELU: {
      current = ReLU<T>::propagate(current);
      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);

      if (training) {
        poolMaxIndices->push_back({});
        current = Pool<T>::propagate(current, pool, poolMaxIndices->back());
      } else {
        std::vector<ulong> unused;
        current = Pool<T>::propagate(current, pool, unused);
      }

      break;
    }

    case LayerType::INSTANCENORM:
    case LayerType::BATCHNORM: {
      const auto& normConfig = std::get<NormLayerConfig>(layerConfig.config);
      NormParameters<T> normParams = this->sharedParams.normParams[normIdx];
      LayerType normType = layerConfig.type;

      std::vector<Tensor3D<T>*> batch = {&current};

      if (training) {
        this->normXNormalized.push_back({});
        this->normStatsMean.push_back({});
        this->normStatsVar.push_back({});
        Normalization<T>::propagate(batch, current.shape, normParams, normConfig, normType, true,
                                    &this->normXNormalized.back(), &this->normStatsMean.back(),
                                    &this->normStatsVar.back());
      } else {
        Normalization<T>::propagate(batch, current.shape, normParams, normConfig, normType, false);
      }

      normIdx++;
      break;
    }

    case LayerType::GLOBALAVGPOOL: {
      GlobalAvgPool<T>::propagate(current, current.shape);
      break;
    }

    case LayerType::GLOBALDUALPOOL: {
      GlobalDualPool<T>::propagate(current, current.shape);
      break;
    }

    case LayerType::FLATTEN: {
      break;
    }

    case LayerType::RESIDUAL_START: {
      residualStack.push(current);
      break;
    }

    case LayerType::RESIDUAL_END: {
      Tensor3D<T> skipInput = residualStack.top();
      residualStack.pop();

      // Determine if projection is needed (channel mismatch)
      bool needsProjection = (skipInput.shape.c != current.shape.c);
      const ResidualProjection<T>* projection = nullptr;

      if (needsProjection) {
        projection = &this->sharedParams.residualParams[residualIdx];
        residualIdx++;
      }

      Residual<T>::propagate(current, skipInput, projection);
      break;
    }
    }
  }

  return current;
}

//===================================================================================================================//

template <typename T>
void CoreCPUWorker<T>::backpropagateCNN(const Tensor3D<T>& dCNNOut, const std::vector<Tensor3D<T>>& intermediates,
                                        const std::vector<std::vector<ulong>>& poolMaxIndices,
                                        std::vector<std::vector<T>>& dConvFilters,
                                        std::vector<std::vector<T>>& dConvBiases, std::vector<std::vector<T>>& dBNGamma,
                                        std::vector<std::vector<T>>& dBNBeta)
{
  ulong numCNNLayers = this->layersConfig.cnnLayers.size();
  ulong numConvLayers = this->sharedParams.convParams.size();
  ulong numBNLayers = this->sharedParams.normParams.size();
  ulong numResidualLayers = this->sharedParams.residualParams.size();

  dConvFilters.resize(numConvLayers);
  dConvBiases.resize(numConvLayers);
  dBNGamma.resize(numBNLayers);
  dBNBeta.resize(numBNLayers);
  this->dResidualWeights.resize(numResidualLayers);
  this->dResidualBiases.resize(numResidualLayers);

  Tensor3D<T> dCurrent = dCNNOut;

  ulong convIdx = numConvLayers;
  ulong poolIdx = poolMaxIndices.size();
  ulong normIdx = numBNLayers;
  ulong residualIdx = numResidualLayers;
  std::stack<Tensor3D<T>> residualGradStack;

  for (long i = static_cast<long>(numCNNLayers) - 1; i >= 0; i--) {
    const CNNLayerConfig& layerConfig = this->layersConfig.cnnLayers[static_cast<ulong>(i)];
    const Tensor3D<T>& layerInput = intermediates[static_cast<ulong>(i)];

    switch (layerConfig.type) {
    case LayerType::CONV: {
      convIdx--;
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      dCurrent = Conv2D<T>::backpropagate(dCurrent, layerInput, conv, this->sharedParams.convParams[convIdx],
                                          dConvFilters[convIdx], dConvBiases[convIdx]);
      break;
    }

    case LayerType::RELU: {
      dCurrent = ReLU<T>::backpropagate(dCurrent, layerInput);
      break;
    }

    case LayerType::POOL: {
      poolIdx--;
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
      dCurrent = Pool<T>::backpropagate(dCurrent, layerInput.shape, pool, poolMaxIndices[poolIdx]);
      break;
    }

    case LayerType::INSTANCENORM:
    case LayerType::BATCHNORM: {
      normIdx--;
      const auto& normConfig = std::get<NormLayerConfig>(layerConfig.config);
      LayerType normType = layerConfig.type;

      std::vector<Tensor3D<T>*> dBatch = {&dCurrent};
      std::vector<T> layerDGamma;
      std::vector<T> layerDBeta;
      Normalization<T>::backpropagate(dBatch, layerInput.shape, this->sharedParams.normParams[normIdx], normConfig,
                                      normType, this->normStatsMean[normIdx], this->normStatsVar[normIdx],
                                      this->normXNormalized[normIdx], layerDGamma, layerDBeta);
      dBNGamma[normIdx] = layerDGamma;
      dBNBeta[normIdx] = layerDBeta;
      break;
    }

    case LayerType::GLOBALAVGPOOL: {
      GlobalAvgPool<T>::backpropagate(dCurrent, layerInput.shape);
      break;
    }

    case LayerType::GLOBALDUALPOOL: {
      GlobalDualPool<T>::backpropagate(dCurrent, layerInput, layerInput.shape);
      break;
    }

    case LayerType::FLATTEN: {
      break;
    }

    case LayerType::RESIDUAL_END: {
      // Going backward: RESIDUAL_END is hit first.
      // The gradient splits: one copy continues through the block (dCurrent unchanged),
      // the other is saved to be added at RESIDUAL_START.
      // We also need the skip input (from intermediates at the matching RESIDUAL_START)
      // to compute projection gradients, but that happens at RESIDUAL_START below.
      residualGradStack.push(dCurrent);
      break;
    }

    case LayerType::RESIDUAL_START: {
      // Going backward: RESIDUAL_START is hit after all block layers have been backpropagated.
      // dCurrent now contains the gradient from the block path.
      // We add the gradient from the skip path (saved at RESIDUAL_END).
      Tensor3D<T> dFromSkip = residualGradStack.top();
      residualGradStack.pop();

      // If there's a projection, backpropagate through it
      bool needsProjection = false;

      // Check if the skip input (layerInput at this RESIDUAL_START) has different channels
      // than the dFromSkip (which has the block output channels)
      if (layerInput.shape.c != dFromSkip.shape.c)
        needsProjection = true;

      if (needsProjection) {
        residualIdx--;
        const ResidualProjection<T>* projection = &this->sharedParams.residualParams[residualIdx];

        ResidualProjection<T> dProj;
        dProj.inC = projection->inC;
        dProj.outC = projection->outC;
        dProj.weights.assign(projection->weights.size(), static_cast<T>(0));
        dProj.biases.assign(projection->biases.size(), static_cast<T>(0));

        Tensor3D<T> dSkip = Residual<T>::backpropagate(dFromSkip, layerInput, projection, &dProj);

        this->dResidualWeights[residualIdx] = std::move(dProj.weights);
        this->dResidualBiases[residualIdx] = std::move(dProj.biases);

        // Add skip gradient to block gradient
        for (ulong j = 0; j < dCurrent.data.size(); j++)
          dCurrent.data[j] += dSkip.data[j];
      } else {
        // Identity shortcut: skip gradient == dFromSkip, just add to dCurrent
        for (ulong j = 0; j < dCurrent.data.size(); j++)
          dCurrent.data[j] += dFromSkip.data[j];
      }

      break;
    }
    }
  }
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::CoreCPUWorker<int>;
template class CNN::CoreCPUWorker<double>;
template class CNN::CoreCPUWorker<float>;
