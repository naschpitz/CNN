#include "CNN_Worker.hpp"
#include "CNN_SlidingStrategy.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <variant>

using namespace CNN;

//===================================================================================================================//

template <typename T>
void Worker<T>::initializeConvParams(const LayersConfig& layersConfig, const Shape3D& inputShape,
                                     Parameters<T>& parameters)
{
  ulong convIdx = 0;
  Shape3D currentShape = inputShape;

  for (const auto& layerConfig : layersConfig.cnnLayers) {
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
    if (convIdx < parameters.convParams.size() && !parameters.convParams[convIdx].filters.empty()) {
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
    if (convIdx >= parameters.convParams.size()) {
      parameters.convParams.resize(convIdx + 1);
    }

    ConvParameters<T>& cp = parameters.convParams[convIdx];
    cp.numFilters = conv.numFilters;
    cp.inputC = currentShape.c;
    cp.filterH = conv.filterH;
    cp.filterW = conv.filterW;

    ulong filterSize = cp.numFilters * cp.inputC * cp.filterH * cp.filterW;
    cp.filters.resize(filterSize);
    cp.biases.assign(cp.numFilters, static_cast<T>(0));

    T fanIn = static_cast<T>(cp.inputC * cp.filterH * cp.filterW);
    T stddev = std::sqrt(static_cast<T>(2) / fanIn);

    std::mt19937 gen(42 + convIdx);
    std::normal_distribution<double> dist(0.0, static_cast<double>(stddev));

    for (ulong i = 0; i < filterSize; i++) {
      cp.filters[i] = static_cast<T>(dist(gen));
    }

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
void Worker<T>::initializeInstanceNormParams(const LayersConfig& layersConfig, const Shape3D& inputShape,
                                             Parameters<T>& parameters)
{
  ulong inIdx = 0;
  Shape3D currentShape = inputShape;

  for (const auto& layerConfig : layersConfig.cnnLayers) {
    switch (layerConfig.type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      currentShape = {conv.numFilters, outH, outW};
      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
      ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
      currentShape = {currentShape.c, outH, outW};
      break;
    }

    case LayerType::INSTANCENORM: {
      ulong numChannels = currentShape.c;

      // Check if parameters already loaded
      if (inIdx < parameters.inParams.size() && !parameters.inParams[inIdx].gamma.empty()) {
        inIdx++;
        break;
      }

      // Ensure inParams vector is large enough
      if (inIdx >= parameters.inParams.size()) {
        parameters.inParams.resize(inIdx + 1);
      }

      InstanceNormParameters<T>& bp = parameters.inParams[inIdx];
      bp.numChannels = numChannels;
      bp.gamma.assign(numChannels, static_cast<T>(1)); // Initialize scale to 1
      bp.beta.assign(numChannels, static_cast<T>(0)); // Initialize shift to 0
      bp.runningMean.assign(numChannels, static_cast<T>(0));
      bp.runningVar.assign(numChannels, static_cast<T>(1));
      inIdx++;
      break;
    }

    case LayerType::RELU:
    case LayerType::FLATTEN:
      break;
    }
  }
}

//===================================================================================================================//

template <typename T>
T Worker<T>::calculateLoss(const Output<T>& predicted, const Output<T>& expected) const
{
  T loss = static_cast<T>(0);

  switch (this->costFunctionConfig.type) {
  case CostFunctionType::CROSS_ENTROPY: {
    // Cross-entropy: L = -sum(w_i * y_i * log(a_i))
    const T epsilon = static_cast<T>(1e-7);

    for (ulong i = 0; i < expected.size(); i++) {
      T pred = std::max(predicted[i], epsilon);
      T weight = (!this->costFunctionConfig.weights.empty()) ? this->costFunctionConfig.weights[i] : static_cast<T>(1);
      loss -= weight * expected[i] * std::log(pred);
    }

    break;
  }

  case CostFunctionType::SQUARED_DIFFERENCE:
  case CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE:
  default: {
    // Squared difference: L = sum(w_i * (a_i - y_i)^2) / N
    for (ulong i = 0; i < expected.size(); i++) {
      T diff = predicted[i] - expected[i];
      T weight = (!this->costFunctionConfig.weights.empty()) ? this->costFunctionConfig.weights[i] : static_cast<T>(1);
      loss += weight * diff * diff;
    }

    loss /= static_cast<T>(expected.size());
    break;
  }
  }

  return loss;
}

//===================================================================================================================//

// Explicit template instantiations.
template class CNN::Worker<int>;
template class CNN::Worker<double>;
template class CNN::Worker<float>;
