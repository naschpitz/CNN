#include "CNN_CoreCPUWorker.hpp"
#include "CNN_Conv2D.hpp"
#include "CNN_ReLU.hpp"
#include "CNN_Pool.hpp"
#include "CNN_Flatten.hpp"
#include "CNN_InstanceNorm.hpp"
#include "CNN_BatchNorm.hpp"

#include <ANN_Core.hpp>

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

    this->accumDNormGamma.resize(sharedParams.normParams.size());
    this->accumDNormBeta.resize(sharedParams.normParams.size());
    this->accumNormMean.resize(sharedParams.normParams.size());
    this->accumNormVar.resize(sharedParams.normParams.size());

    for (ulong i = 0; i < sharedParams.normParams.size(); i++) {
      this->accumDNormGamma[i].resize(sharedParams.normParams[i].numChannels, static_cast<T>(0));
      this->accumDNormBeta[i].resize(sharedParams.normParams[i].numChannels, static_cast<T>(0));
      this->accumNormMean[i].resize(sharedParams.normParams[i].numChannels, static_cast<T>(0));
      this->accumNormVar[i].resize(sharedParams.normParams[i].numChannels, static_cast<T>(0));
    }

    this->normSampleCount = 0;
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
  annConfig.seed = cnnConfig.seed;

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
  std::vector<std::vector<T>> dNormGamma, dNormBeta;
  this->backpropagateCNN(dCNNOut, intermediates, poolMaxIndices, dConvFilters, dConvBiases, dNormGamma, dNormBeta);

  // Accumulate CNN gradients
  for (ulong i = 0; i < dConvFilters.size(); i++) {
    for (ulong j = 0; j < dConvFilters[i].size(); j++)
      this->accumDConvFilters[i][j] += dConvFilters[i][j];

    for (ulong j = 0; j < dConvBiases[i].size(); j++)
      this->accumDConvBiases[i][j] += dConvBiases[i][j];
  }

  // Accumulate norm gradients and running stats
  for (ulong i = 0; i < dNormGamma.size(); i++) {
    for (ulong j = 0; j < dNormGamma[i].size(); j++)
      this->accumDNormGamma[i][j] += dNormGamma[i][j];

    for (ulong j = 0; j < dNormBeta[i].size(); j++)
      this->accumDNormBeta[i][j] += dNormBeta[i][j];

    for (ulong j = 0; j < this->normBatchMeans[i].size(); j++)
      this->accumNormMean[i][j] += this->normBatchMeans[i][j];

    for (ulong j = 0; j < this->normBatchVars[i].size(); j++)
      this->accumNormVar[i][j] += this->normBatchVars[i][j];
  }

  this->normSampleCount++;
  this->accum_loss += sampleLoss;

  return sampleLoss;
}

//===================================================================================================================//
//-- processBatch: layer-by-layer forward/backward for true BatchNorm --//
//===================================================================================================================//

template <typename T>
T CoreCPUWorker<T>::processBatch(const std::vector<std::pair<Input<T>, Output<T>>>& batch)
{
  ulong N = batch.size();
  ulong numCNNLayers = this->layersConfig.cnnLayers.size();

  // ---- Forward pass: layer by layer ----

  // batchActvs[sampleIdx] = current activation for that sample
  std::vector<Tensor3D<T>> batchActvs(N);

  for (ulong s = 0; s < N; s++)
    batchActvs[s] = batch[s].first;

  // intermediates[sampleIdx][layerIdx] = activation BEFORE that layer (needed for backprop)
  std::vector<std::vector<Tensor3D<T>>> allIntermediates(N);
  std::vector<std::vector<std::vector<ulong>>> allPoolMaxIndices(N);

  for (ulong s = 0; s < N; s++) {
    allIntermediates[s].reserve(numCNNLayers);
    allPoolMaxIndices[s].reserve(numCNNLayers);
  }

  // Clear norm intermediates
  this->trueBNMeans.clear();
  this->trueBNVars.clear();
  this->trueBNXNorm.clear();
  this->normBatchMeans.clear();
  this->normBatchVars.clear();
  this->normXNormalized.clear();

  ulong convIdx = 0;
  ulong normIdx = 0;

  for (ulong l = 0; l < numCNNLayers; l++) {
    const auto& layerConfig = this->layersConfig.cnnLayers[l];

    // Save intermediates for all samples
    for (ulong s = 0; s < N; s++)
      allIntermediates[s].push_back(batchActvs[s]);

    switch (layerConfig.type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);

      for (ulong s = 0; s < N; s++)
        batchActvs[s] = Conv2D<T>::propagate(batchActvs[s], conv, this->sharedParams.convParams[convIdx]);

      convIdx++;
      break;
    }

    case LayerType::RELU: {
      for (ulong s = 0; s < N; s++)
        batchActvs[s] = ReLU<T>::propagate(batchActvs[s]);

      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);

      for (ulong s = 0; s < N; s++) {
        allPoolMaxIndices[s].push_back({});
        batchActvs[s] = Pool<T>::propagate(batchActvs[s], pool, allPoolMaxIndices[s].back());
      }

      break;
    }

    case LayerType::BATCHNORM: {
      // TRUE batch normalization: compute stats across all N samples
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      NormParameters<T> normParams = this->sharedParams.normParams[normIdx];
      Shape3D shape = batchActvs[0].shape;

      this->trueBNMeans.push_back({});
      this->trueBNVars.push_back({});
      this->trueBNXNorm.push_back({});

      batchActvs = BatchNorm<T>::propagate(batchActvs, shape, normParams, bn, &this->trueBNMeans.back(),
                                           &this->trueBNVars.back(), &this->trueBNXNorm.back());
      normIdx++;
      break;
    }

    case LayerType::INSTANCENORM: {
      const auto& in = std::get<NormLayerConfig>(layerConfig.config);
      NormParameters<T> normParams = this->sharedParams.normParams[normIdx];

      for (ulong s = 0; s < N; s++) {
        this->normBatchMeans.push_back({});
        this->normBatchVars.push_back({});
        this->normXNormalized.push_back({});
        batchActvs[s] =
          InstanceNorm<T>::propagate(batchActvs[s], batchActvs[s].shape, normParams, in, &this->normBatchMeans.back(),
                                     &this->normBatchVars.back(), &this->normXNormalized.back());
      }

      normIdx++;
      break;
    }

    case LayerType::FLATTEN: {
      break;
    }
    }
  }

  // ---- ANN forward + loss + ANN backward for each sample ----
  T batchLoss = static_cast<T>(0);
  std::vector<Tensor3D<T>> dCNNOuts(N);

  for (ulong s = 0; s < N; s++) {
    Tensor1D<T> flatInput = Flatten<T>::propagate(batchActvs[s]);

    // ANN forward
    ANN::Input<T> annInput(flatInput.begin(), flatInput.end());
    ANN::Output<T> annOutput = this->annCore->predict(annInput);
    Output<T> predicted(annOutput.begin(), annOutput.end());

    // Loss
    T sampleLoss = this->calculateLoss(predicted, batch[s].second);
    batchLoss += sampleLoss;

    // ANN backward + accumulate
    ANN::Output<T> annExpected(batch[s].second.begin(), batch[s].second.end());
    ANN::Tensor1D<T> dFlatInput = this->annCore->backpropagate(annExpected);
    this->annCore->accumulate();

    Tensor1D<T> dFlat(dFlatInput.begin(), dFlatInput.end());
    dCNNOuts[s] = Flatten<T>::backpropagate(dFlat, this->cnnOutputShape);
  }

  // ---- CNN backward pass: layer by layer (reversed) ----

  ulong numConvLayers = this->sharedParams.convParams.size();
  ulong numNormLayers = this->sharedParams.normParams.size();

  std::vector<std::vector<T>> dConvFilters(numConvLayers);
  std::vector<std::vector<T>> dConvBiases(numConvLayers);
  std::vector<std::vector<T>> dNormGamma(numNormLayers);
  std::vector<std::vector<T>> dNormBeta(numNormLayers);

  convIdx = numConvLayers;
  normIdx = numNormLayers;
  ulong poolIdx = 0;
  ulong bnIntermIdx = this->trueBNMeans.size(); // Reverse index for trueBN intermediates
  ulong inIntermIdx = this->normBatchMeans.size(); // Reverse index for IN intermediates

  // Count pool layers for reverse indexing
  for (ulong s = 0; s < N; s++)
    poolIdx = allPoolMaxIndices[s].size();

  for (long i = static_cast<long>(numCNNLayers) - 1; i >= 0; i--) {
    const auto& layerConfig = this->layersConfig.cnnLayers[static_cast<ulong>(i)];

    switch (layerConfig.type) {
    case LayerType::CONV: {
      convIdx--;
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);

      for (ulong s = 0; s < N; s++) {
        std::vector<T> sampleDF, sampleDB;
        dCNNOuts[s] = Conv2D<T>::backpropagate(dCNNOuts[s], allIntermediates[s][static_cast<ulong>(i)], conv,
                                               this->sharedParams.convParams[convIdx], sampleDF, sampleDB);

        if (s == 0) {
          dConvFilters[convIdx] = std::move(sampleDF);
          dConvBiases[convIdx] = std::move(sampleDB);
        } else {
          for (ulong j = 0; j < dConvFilters[convIdx].size(); j++)
            dConvFilters[convIdx][j] += sampleDF[j];

          for (ulong j = 0; j < dConvBiases[convIdx].size(); j++)
            dConvBiases[convIdx][j] += sampleDB[j];
        }
      }

      break;
    }

    case LayerType::RELU: {
      for (ulong s = 0; s < N; s++)
        dCNNOuts[s] = ReLU<T>::backpropagate(dCNNOuts[s], allIntermediates[s][static_cast<ulong>(i)]);

      break;
    }

    case LayerType::POOL: {
      poolIdx--;
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);

      for (ulong s = 0; s < N; s++)
        dCNNOuts[s] = Pool<T>::backpropagate(dCNNOuts[s], allIntermediates[s][static_cast<ulong>(i)].shape, pool,
                                             allPoolMaxIndices[s][poolIdx]);

      break;
    }

    case LayerType::BATCHNORM: {
      normIdx--;
      bnIntermIdx--;
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);

      // True batch-wide backpropagation
      dCNNOuts = BatchNorm<T>::backpropagate(dCNNOuts, allIntermediates[0][static_cast<ulong>(i)].shape,
                                             this->sharedParams.normParams[normIdx], bn, this->trueBNMeans[bnIntermIdx],
                                             this->trueBNVars[bnIntermIdx], this->trueBNXNorm[bnIntermIdx],
                                             dNormGamma[normIdx], dNormBeta[normIdx]);
      break;
    }

    case LayerType::INSTANCENORM: {
      normIdx--;
      const auto& in = std::get<NormLayerConfig>(layerConfig.config);

      for (long s = static_cast<long>(N) - 1; s >= 0; s--) {
        inIntermIdx--;
        std::vector<T> sDG, sDB;
        dCNNOuts[static_cast<ulong>(s)] = InstanceNorm<T>::backpropagate(
          dCNNOuts[static_cast<ulong>(s)], allIntermediates[static_cast<ulong>(s)][static_cast<ulong>(i)].shape,
          this->sharedParams.normParams[normIdx], in, this->normBatchMeans[inIntermIdx],
          this->normBatchVars[inIntermIdx], this->normXNormalized[inIntermIdx], sDG, sDB);

        if (s == static_cast<long>(N) - 1) {
          dNormGamma[normIdx] = std::move(sDG);
          dNormBeta[normIdx] = std::move(sDB);
        } else {
          for (ulong j = 0; j < dNormGamma[normIdx].size(); j++)
            dNormGamma[normIdx][j] += sDG[j];

          for (ulong j = 0; j < dNormBeta[normIdx].size(); j++)
            dNormBeta[normIdx][j] += sDB[j];
        }
      }

      break;
    }

    case LayerType::FLATTEN: {
      break;
    }
    }
  }

  // ---- Accumulate gradients ----
  for (ulong i = 0; i < dConvFilters.size(); i++) {
    for (ulong j = 0; j < dConvFilters[i].size(); j++)
      this->accumDConvFilters[i][j] += dConvFilters[i][j];

    for (ulong j = 0; j < dConvBiases[i].size(); j++)
      this->accumDConvBiases[i][j] += dConvBiases[i][j];
  }

  for (ulong i = 0; i < dNormGamma.size(); i++) {
    for (ulong j = 0; j < dNormGamma[i].size(); j++)
      this->accumDNormGamma[i][j] += dNormGamma[i][j];

    for (ulong j = 0; j < dNormBeta[i].size(); j++)
      this->accumDNormBeta[i][j] += dNormBeta[i][j];
  }

  // Running stats (runningMean, runningVar) are updated directly inside
  // BatchNorm::propagate() and InstanceNorm::propagate() during the forward pass.
  this->accum_loss += batchLoss;

  return batchLoss;
}

//===================================================================================================================//

template <typename T>
void CoreCPUWorker<T>::resetAccumulators()
{
  for (ulong i = 0; i < this->accumDConvFilters.size(); i++) {
    std::fill(this->accumDConvFilters[i].begin(), this->accumDConvFilters[i].end(), static_cast<T>(0));
    std::fill(this->accumDConvBiases[i].begin(), this->accumDConvBiases[i].end(), static_cast<T>(0));
  }

  for (ulong i = 0; i < this->accumDNormGamma.size(); i++) {
    std::fill(this->accumDNormGamma[i].begin(), this->accumDNormGamma[i].end(), static_cast<T>(0));
    std::fill(this->accumDNormBeta[i].begin(), this->accumDNormBeta[i].end(), static_cast<T>(0));
    std::fill(this->accumNormMean[i].begin(), this->accumNormMean[i].end(), static_cast<T>(0));
    std::fill(this->accumNormVar[i].begin(), this->accumNormVar[i].end(), static_cast<T>(0));
  }

  this->normSampleCount = 0;
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
    this->normBatchMeans.clear();
    this->normBatchVars.clear();
    this->normXNormalized.clear();
  }

  Tensor3D<T> current = input;
  ulong convIdx = 0;
  ulong normIdx = 0;

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

    case LayerType::BATCHNORM: {
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      NormParameters<T> normParams = this->sharedParams.normParams[normIdx];

      if (training) {
        // TODO: Phase 2 — true batch-wide processing. For now, uses InstanceNorm per-sample.
        this->normBatchMeans.push_back({});
        this->normBatchVars.push_back({});
        this->normXNormalized.push_back({});
        current = InstanceNorm<T>::propagate(current, current.shape, normParams, bn, &this->normBatchMeans.back(),
                                             &this->normBatchVars.back(), &this->normXNormalized.back());
      } else {
        // Inference: single sample, no intermediates → uses running stats
        current = BatchNorm<T>::propagate({current}, current.shape, normParams, bn)[0];
      }

      normIdx++;
      break;
    }

    case LayerType::INSTANCENORM: {
      const auto& in = std::get<NormLayerConfig>(layerConfig.config);
      NormParameters<T> normParams = this->sharedParams.normParams[normIdx];

      if (training) {
        this->normBatchMeans.push_back({});
        this->normBatchVars.push_back({});
        this->normXNormalized.push_back({});
        current = InstanceNorm<T>::propagate(current, current.shape, normParams, in, &this->normBatchMeans.back(),
                                             &this->normBatchVars.back(), &this->normXNormalized.back());
      } else {
        current = InstanceNorm<T>::propagate(current, current.shape, normParams, in);
      }

      normIdx++;
      break;
    }

    case LayerType::FLATTEN: {
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
                                        std::vector<std::vector<T>>& dConvBiases,
                                        std::vector<std::vector<T>>& dNormGamma, std::vector<std::vector<T>>& dNormBeta)
{
  ulong numCNNLayers = this->layersConfig.cnnLayers.size();
  ulong numConvLayers = this->sharedParams.convParams.size();
  ulong numNormLayers = this->sharedParams.normParams.size();

  dConvFilters.resize(numConvLayers);
  dConvBiases.resize(numConvLayers);
  dNormGamma.resize(numNormLayers);
  dNormBeta.resize(numNormLayers);

  Tensor3D<T> dCurrent = dCNNOut;

  ulong convIdx = numConvLayers;
  ulong poolIdx = poolMaxIndices.size();
  ulong normIdx = numNormLayers;

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

    case LayerType::BATCHNORM:
    case LayerType::INSTANCENORM: {
      normIdx--;
      const auto& normConfig = std::get<NormLayerConfig>(layerConfig.config);
      // Per-sample backprop (used in single-sample processSample path)
      dCurrent = InstanceNorm<T>::backpropagate(
        dCurrent, layerInput.shape, this->sharedParams.normParams[normIdx], normConfig, this->normBatchMeans[normIdx],
        this->normBatchVars[normIdx], this->normXNormalized[normIdx], dNormGamma[normIdx], dNormBeta[normIdx]);
      break;
    }

    case LayerType::FLATTEN: {
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
