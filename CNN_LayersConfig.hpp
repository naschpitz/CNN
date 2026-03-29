#ifndef CNN_LAYERSCONFIG_HPP
#define CNN_LAYERSCONFIG_HPP

#include "CNN_SlidingStrategy.hpp"
#include "CNN_PoolType.hpp"
#include "CNN_Types.hpp"

#include <ANN_ActvFunc.hpp>

#include <string>
#include <variant>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  // Convolution layer configuration
  struct ConvLayerConfig {
      ulong numFilters;
      ulong filterH;
      ulong filterW;
      ulong strideY;
      ulong strideX;
      SlidingStrategyType slidingStrategy;
  };

  // ReLU layer configuration (no parameters)
  struct ReLULayerConfig {
  };

  // Pooling layer configuration
  struct PoolLayerConfig {
      PoolTypeEnum poolType;
      ulong poolH;
      ulong poolW;
      ulong strideY;
      ulong strideX;
  };

  // Flatten layer configuration (no parameters, auto-inserted before dense)
  struct FlattenLayerConfig {
  };

  // Global average pooling layer configuration (no parameters)
  struct GlobalAvgPoolLayerConfig {
  };

  // Global dual pooling layer configuration (avg + max concatenated, no parameters)
  struct GlobalDualPoolLayerConfig {
  };

  // Batch normalization layer configuration
  struct NormLayerConfig {
      float epsilon = 1e-5f; // Small constant for numerical stability
      float momentum = 0.1f; // Momentum for running mean/variance update
  };

  // Residual block markers (no parameters — projection weights handled separately)
  struct ResidualStartConfig {
  };

  struct ResidualEndConfig {
  };

  // A CNN layer can be any of these types
  enum class LayerType {
    CONV,
    RELU,
    POOL,
    FLATTEN,
    GLOBALAVGPOOL,
    GLOBALDUALPOOL,
    INSTANCENORM,
    BATCHNORM,
    RESIDUAL_START,
    RESIDUAL_END
  };

  struct CNNLayerConfig {
      LayerType type;
      std::variant<ConvLayerConfig, ReLULayerConfig, PoolLayerConfig, FlattenLayerConfig, GlobalAvgPoolLayerConfig,
                   GlobalDualPoolLayerConfig, NormLayerConfig, ResidualStartConfig, ResidualEndConfig>
        config;
  };

  // Dense layer configuration (delegates to ANN)
  struct DenseLayerConfig {
      ulong numNeurons;
      ANN::ActvFuncType actvFuncType;
  };

  // Full CNN layers configuration
  class LayersConfig
  {
    public:
      std::vector<CNNLayerConfig> cnnLayers;
      std::vector<DenseLayerConfig> denseLayers;

      // Validate layer-by-layer shape compatibility
      // Returns the output shape of the CNN portion (before flatten)
      Shape3D validateShapes(const Shape3D& inputShape) const;
  };
}

//===================================================================================================================//

#endif // CNN_LAYERSCONFIG_HPP
