#include "CNN_CoreCPU.hpp"
#include "CNN_Conv2D.hpp"
#include "CNN_ReLU.hpp"
#include "CNN_Pool.hpp"
#include "CNN_Flatten.hpp"

#include <ANN_Core.hpp>
#include <ANN_ActvFunc.hpp>

#include <QDebug>

#include <cmath>
#include <random>
#include <stdexcept>

using namespace CNN;

//===================================================================================================================//

template <typename T>
CoreCPU<T>::CoreCPU(const CoreConfig<T>& config) : Core<T>(config) {
  // Compute CNN output shape (before flatten)
  cnnOutputShape = this->layersConfig.validateShapes(this->inputShape);
  flattenSize = cnnOutputShape.size();

  // Initialize conv parameters if not loaded
  initializeConvParams();

  // Build and create ANN core for dense layers
  ANN::CoreConfig<T> annConfig = buildANNConfig(config);
  annCore = ANN::Core<T>::makeCore(annConfig);

  // Initialize CNN gradient accumulators
  accumDConvFilters.resize(this->parameters.convParams.size());
  accumDConvBiases.resize(this->parameters.convParams.size());

  for (ulong i = 0; i < this->parameters.convParams.size(); i++) {
    accumDConvFilters[i].resize(this->parameters.convParams[i].filters.size(), static_cast<T>(0));
    accumDConvBiases[i].resize(this->parameters.convParams[i].biases.size(), static_cast<T>(0));
  }
}

//===================================================================================================================//

template <typename T>
ANN::CoreConfig<T> CoreCPU<T>::buildANNConfig(const CoreConfig<T>& cnnConfig) {
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

  // Input layer (flatten output size) - use identity activation (not used in forward pass per ANN convention)
  ANN::Layer inputLayer;
  inputLayer.numNeurons = flattenSize;
  // ANN's first layer is input-only: its activation is never used in propagate().
  // Use RELU as a harmless placeholder (ANN has no IDENTITY type).
  inputLayer.actvFuncType = ANN::ActvFuncType::RELU;
  annLayers.push_back(inputLayer);

  // Hidden/output layers from CNN's dense config
  for (const auto& denseConfig : cnnConfig.layersConfig.denseLayers) {
    ANN::Layer layer;
    layer.numNeurons = denseConfig.numNeurons;
    layer.actvFuncType = denseConfig.actvFuncType;
    annLayers.push_back(layer);
  }

  annConfig.layersConfig = annLayers;

  // Training config
  annConfig.trainingConfig.numEpochs = cnnConfig.trainingConfig.numEpochs;
  annConfig.trainingConfig.learningRate = cnnConfig.trainingConfig.learningRate;
  annConfig.trainingConfig.numThreads = 1; // CNN manages its own threading

  // Dense parameters (if loaded from file)
  annConfig.parameters = cnnConfig.parameters.denseParams;

  annConfig.verbose = cnnConfig.verbose;

  return annConfig;
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::initializeConvParams() {
  ulong convIdx = 0;
  Shape3D currentShape = this->inputShape;

  for (const auto& layerConfig : this->layersConfig.cnnLayers) {
    if (layerConfig.type != LayerType::CONV) {
      // Update shape for non-conv layers
      if (layerConfig.type == LayerType::POOL) {
        const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
        ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
        ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
        currentShape = {currentShape.c, outH, outW};
      }
      // ReLU and Flatten don't change shape (Flatten is at end)
      continue;
    }

    const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);

    // Check if parameters already loaded
    if (convIdx < this->parameters.convParams.size() &&
        !this->parameters.convParams[convIdx].filters.empty()) {
      // Parameters already loaded - update shape and continue
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      currentShape = {conv.numFilters, outH, outW};
      convIdx++;
      continue;
    }

    // Ensure convParams vector is large enough
    if (convIdx >= this->parameters.convParams.size()) {
      this->parameters.convParams.resize(convIdx + 1);
    }

    ConvParameters<T>& cp = this->parameters.convParams[convIdx];
    cp.numFilters = conv.numFilters;
    cp.inputC = currentShape.c;
    cp.filterH = conv.filterH;
    cp.filterW = conv.filterW;

    ulong filterSize = cp.numFilters * cp.inputC * cp.filterH * cp.filterW;
    cp.filters.resize(filterSize);
    cp.biases.assign(cp.numFilters, static_cast<T>(0));

    // He initialization: stddev = sqrt(2 / fan_in), fan_in = inputC * filterH * filterW
    T fanIn = static_cast<T>(cp.inputC * cp.filterH * cp.filterW);
    T stddev = std::sqrt(static_cast<T>(2) / fanIn);

    std::mt19937 gen(42 + convIdx); // Deterministic seed per layer
    std::normal_distribution<double> dist(0.0, static_cast<double>(stddev));

    for (ulong i = 0; i < filterSize; i++) {
      cp.filters[i] = static_cast<T>(dist(gen));
    }

    // Update shape for next layer
    ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
    ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
    ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
    ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
    currentShape = {conv.numFilters, outH, outW};

    convIdx++;
  }
}

//===================================================================================================================//

template <typename T>
Tensor3D<T> CoreCPU<T>::forwardCNN(const Input<T>& input) {
  Tensor3D<T> current = input;
  ulong convIdx = 0;

  for (const auto& layerConfig : this->layersConfig.cnnLayers) {
    switch (layerConfig.type) {
      case LayerType::CONV: {
        const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
        current = Conv2D<T>::predict(current, conv, this->parameters.convParams[convIdx]);
        convIdx++;
        break;
      }
      case LayerType::RELU: {
        current = ReLU<T>::predict(current);
        break;
      }
      case LayerType::POOL: {
        const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
        std::vector<ulong> unused;
        current = Pool<T>::predict(current, pool, unused);
        break;
      }
      case LayerType::FLATTEN: {
        // Flatten is handled separately after CNN layers
        break;
      }
    }
  }

  return current;
}

//===================================================================================================================//

template <typename T>
Tensor3D<T> CoreCPU<T>::forwardCNN(const Input<T>& input,
                                   std::vector<Tensor3D<T>>& intermediates,
                                   std::vector<std::vector<ulong>>& poolMaxIndices) {
  intermediates.clear();
  poolMaxIndices.clear();

  Tensor3D<T> current = input;
  ulong convIdx = 0;
  ulong poolIdx = 0;

  for (const auto& layerConfig : this->layersConfig.cnnLayers) {
    // Store input to this layer (for backprop)
    intermediates.push_back(current);

    switch (layerConfig.type) {
      case LayerType::CONV: {
        const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
        current = Conv2D<T>::predict(current, conv, this->parameters.convParams[convIdx]);
        convIdx++;
        break;
      }
      case LayerType::RELU: {
        current = ReLU<T>::predict(current);
        break;
      }
      case LayerType::POOL: {
        const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
        poolMaxIndices.push_back({});
        current = Pool<T>::predict(current, pool, poolMaxIndices.back());
        poolIdx++;
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
Output<T> CoreCPU<T>::predict(const Input<T>& input) {
  // Forward through CNN layers
  Tensor3D<T> cnnOut = forwardCNN(input);

  // Flatten
  Tensor1D<T> flatInput = Flatten<T>::predict(cnnOut);

  // Forward through ANN dense layers
  ANN::Input<T> annInput(flatInput.begin(), flatInput.end());
  ANN::Output<T> annOutput = annCore->predict(annInput);

  return Output<T>(annOutput.begin(), annOutput.end());
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::backwardCNN(const Tensor3D<T>& dCNNOut,
                             const std::vector<Tensor3D<T>>& intermediates,
                             const std::vector<std::vector<ulong>>& poolMaxIndices,
                             std::vector<std::vector<T>>& dConvFilters,
                             std::vector<std::vector<T>>& dConvBiases) {
  ulong numCNNLayers = this->layersConfig.cnnLayers.size();
  ulong numConvLayers = this->parameters.convParams.size();

  dConvFilters.resize(numConvLayers);
  dConvBiases.resize(numConvLayers);

  Tensor3D<T> dCurrent = dCNNOut;

  // Count conv and pool layers for reverse indexing
  ulong convIdx = numConvLayers;
  ulong poolIdx = poolMaxIndices.size();

  // Backward pass through CNN layers in reverse
  for (long i = static_cast<long>(numCNNLayers) - 1; i >= 0; i--) {
    const CNNLayerConfig& layerConfig = this->layersConfig.cnnLayers[static_cast<ulong>(i)];
    const Tensor3D<T>& layerInput = intermediates[static_cast<ulong>(i)];

    switch (layerConfig.type) {
      case LayerType::CONV: {
        convIdx--;
        const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
        dCurrent = Conv2D<T>::backpropagate(dCurrent, layerInput, conv,
                                            this->parameters.convParams[convIdx],
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
      case LayerType::FLATTEN: {
        // Flatten layer doesn't have its own backprop in the CNN loop
        // (handled by unflatten before entering backwardCNN)
        break;
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
T CoreCPU<T>::calculateLoss(const Output<T>& predicted, const Output<T>& expected) {
  T loss = static_cast<T>(0);

  for (ulong i = 0; i < expected.size(); i++) {
    T diff = predicted[i] - expected[i];
    loss += diff * diff;
  }

  return loss / static_cast<T>(expected.size());
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::resetCNNAccumulators() {
  for (ulong i = 0; i < accumDConvFilters.size(); i++) {
    std::fill(accumDConvFilters[i].begin(), accumDConvFilters[i].end(), static_cast<T>(0));
    std::fill(accumDConvBiases[i].begin(), accumDConvBiases[i].end(), static_cast<T>(0));
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::accumulateCNNGradients(const std::vector<std::vector<T>>& dConvFilters,
                                        const std::vector<std::vector<T>>& dConvBiases) {
  for (ulong i = 0; i < dConvFilters.size(); i++) {
    for (ulong j = 0; j < dConvFilters[i].size(); j++) {
      accumDConvFilters[i][j] += dConvFilters[i][j];
    }

    for (ulong j = 0; j < dConvBiases[i].size(); j++) {
      accumDConvBiases[i][j] += dConvBiases[i][j];
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::updateCNNParameters(ulong numSamples) {
  T lr = static_cast<T>(this->trainingConfig.learningRate);
  T n = static_cast<T>(numSamples);

  for (ulong i = 0; i < this->parameters.convParams.size(); i++) {
    for (ulong j = 0; j < this->parameters.convParams[i].filters.size(); j++) {
      this->parameters.convParams[i].filters[j] -= lr * (accumDConvFilters[i][j] / n);
    }

    for (ulong j = 0; j < this->parameters.convParams[i].biases.size(); j++) {
      this->parameters.convParams[i].biases[j] -= lr * (accumDConvBiases[i][j] / n);
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::train(const Samples<T>& samples) {
  ulong numEpochs = this->trainingConfig.numEpochs;
  ulong numSamples = samples.size();

  if (numSamples == 0) {
    throw std::runtime_error("No training samples provided");
  }

  this->trainingStart(numSamples);

  if (this->verbose) {
    qDebug() << "CNN Training: " << numEpochs << " epochs, " << numSamples << " samples";
  }

  ulong progressInterval = (this->progressReports > 0)
    ? std::max(static_cast<ulong>(1), numSamples / this->progressReports)
    : 0;

  for (ulong e = 0; e < numEpochs; e++) {
    // Reset accumulators for this epoch
    resetCNNAccumulators();
    annCore->resetAccumulators();

    T epochLoss = static_cast<T>(0);

    for (ulong s = 0; s < numSamples; s++) {
      const Sample<T>& sample = samples[s];

      // 1. Forward through CNN layers (with intermediates for backprop)
      std::vector<Tensor3D<T>> intermediates;
      std::vector<std::vector<ulong>> poolMaxIndices;
      Tensor3D<T> cnnOut = forwardCNN(sample.input, intermediates, poolMaxIndices);

      // 2. Flatten CNN output
      Tensor1D<T> flatInput = Flatten<T>::predict(cnnOut);

      // 3. Forward through ANN
      ANN::Input<T> annInput(flatInput.begin(), flatInput.end());
      ANN::Output<T> annOutput = annCore->predict(annInput);

      // 4. Calculate loss
      Output<T> predicted(annOutput.begin(), annOutput.end());
      T sampleLoss = calculateLoss(predicted, sample.output);
      epochLoss += sampleLoss;

      // 5. ANN backpropagation - returns dCost/dInput (gradient w.r.t. flatten output)
      ANN::Output<T> annExpected(sample.output.begin(), sample.output.end());
      ANN::Tensor1D<T> dFlatInput = annCore->backpropagate(annExpected);

      // 6. ANN accumulate gradients
      annCore->accumulate();

      // 7. Unflatten gradient back to 3D
      Tensor1D<T> dFlat(dFlatInput.begin(), dFlatInput.end());
      Tensor3D<T> dCNNOut = Flatten<T>::backpropagate(dFlat, cnnOutputShape);

      // 8. Backward through CNN layers
      std::vector<std::vector<T>> dConvFilters, dConvBiases;
      backwardCNN(dCNNOut, intermediates, poolMaxIndices, dConvFilters, dConvBiases);

      // 9. Accumulate CNN gradients
      accumulateCNNGradients(dConvFilters, dConvBiases);

      // Report progress
      if (progressInterval > 0 && this->trainingCallback && ((s + 1) % progressInterval == 0 || s == numSamples - 1)) {
        TrainingProgress<T> progress;
        progress.currentEpoch = e + 1;
        progress.totalEpochs = numEpochs;
        progress.currentSample = s + 1;
        progress.totalSamples = numSamples;
        progress.sampleLoss = sampleLoss;
        progress.epochLoss = epochLoss / static_cast<T>(s + 1);
        this->trainingCallback(progress);
      }
    }

    // Update weights for this epoch
    annCore->update(numSamples);
    updateCNNParameters(numSamples);

    T avgLoss = epochLoss / static_cast<T>(numSamples);
    this->trainingMetadata.finalLoss = avgLoss;

    if (this->verbose) {
      qDebug() << "Epoch " << (e + 1) << "/" << numEpochs << " - Loss: " << avgLoss;
    }
  }

  this->trainingEnd();

  // Sync ANN parameters back to CNN's parameters struct
  this->parameters.denseParams = annCore->getParameters();
}

//===================================================================================================================//

template <typename T>
TestResult<T> CoreCPU<T>::test(const Samples<T>& samples) {
  TestResult<T> result;
  result.numSamples = samples.size();
  result.totalLoss = static_cast<T>(0);

  for (ulong i = 0; i < samples.size(); i++) {
    Output<T> predicted = predict(samples[i].input);
    result.totalLoss += calculateLoss(predicted, samples[i].output);
  }

  result.averageLoss = (result.numSamples > 0)
    ? result.totalLoss / static_cast<T>(result.numSamples)
    : static_cast<T>(0);

  return result;
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::CoreCPU<int>;
template class CNN::CoreCPU<double>;
template class CNN::CoreCPU<float>;