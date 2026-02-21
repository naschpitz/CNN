#include "CNN_LayersConfig.hpp"

#include <stdexcept>
#include <string>

using namespace CNN;

//===================================================================================================================//

Shape3D LayersConfig::validateShapes(const Shape3D& inputShape) const {
  Shape3D current = inputShape;

  for (ulong i = 0; i < cnnLayers.size(); i++) {
    const CNNLayerConfig& layerConfig = cnnLayers[i];

    switch (layerConfig.type) {
      case LayerType::CONV: {
        const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);

        ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
        ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);

        // Check that padded input is large enough for the filter
        if (current.h + 2 * padY < conv.filterH) {
          throw std::runtime_error("CNN layer " + std::to_string(i) + " (conv): "
            "input height (" + std::to_string(current.h) + ") + 2*padY (" + std::to_string(2 * padY) +
            ") < filterH (" + std::to_string(conv.filterH) + ")");
        }

        if (current.w + 2 * padX < conv.filterW) {
          throw std::runtime_error("CNN layer " + std::to_string(i) + " (conv): "
            "input width (" + std::to_string(current.w) + ") + 2*padX (" + std::to_string(2 * padX) +
            ") < filterW (" + std::to_string(conv.filterW) + ")");
        }

        ulong outH = (current.h + 2 * padY - conv.filterH) / conv.strideY + 1;
        ulong outW = (current.w + 2 * padX - conv.filterW) / conv.strideX + 1;

        if (outH == 0 || outW == 0) {
          throw std::runtime_error("CNN layer " + std::to_string(i) + " (conv): "
            "output size is zero. Input: " + std::to_string(current.h) + "x" + std::to_string(current.w) +
            ", filter: " + std::to_string(conv.filterH) + "x" + std::to_string(conv.filterW) +
            ", stride: " + std::to_string(conv.strideY) + "x" + std::to_string(conv.strideX) +
            ", padding: " + std::to_string(padY) + "x" + std::to_string(padX));
        }

        current = {conv.numFilters, outH, outW};
        break;
      }

      case LayerType::RELU: {
        // ReLU does not change shape
        break;
      }

      case LayerType::POOL: {
        const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);

        if (current.h < pool.poolH) {
          throw std::runtime_error("CNN layer " + std::to_string(i) + " (pool): "
            "input height (" + std::to_string(current.h) + ") < poolH (" + std::to_string(pool.poolH) + ")");
        }

        if (current.w < pool.poolW) {
          throw std::runtime_error("CNN layer " + std::to_string(i) + " (pool): "
            "input width (" + std::to_string(current.w) + ") < poolW (" + std::to_string(pool.poolW) + ")");
        }

        ulong outH = (current.h - pool.poolH) / pool.strideY + 1;
        ulong outW = (current.w - pool.poolW) / pool.strideX + 1;

        if (outH == 0 || outW == 0) {
          throw std::runtime_error("CNN layer " + std::to_string(i) + " (pool): "
            "output size is zero. Input: " + std::to_string(current.h) + "x" + std::to_string(current.w) +
            ", pool: " + std::to_string(pool.poolH) + "x" + std::to_string(pool.poolW) +
            ", stride: " + std::to_string(pool.strideY) + "x" + std::to_string(pool.strideX));
        }

        current = {current.c, outH, outW};
        break;
      }

      case LayerType::FLATTEN: {
        // Flatten converts 3D to 1D - shape check is deferred
        break;
      }
    }
  }

  return current;
}

