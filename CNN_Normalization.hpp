#ifndef CNN_NORMALIZATION_HPP
#define CNN_NORMALIZATION_HPP

#include "CNN_Types.hpp"
#include "CNN_NormParameters.hpp"
#include "CNN_LayersConfig.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class Normalization
  {
    public:
      //-- Forward pass --//

      // Unified forward pass for both InstanceNorm and BatchNorm.
      // normType controls the normalization scope:
      //   INSTANCENORM: stats computed per-sample over (H,W) — each sample normalized independently
      //   BATCHNORM:    stats computed across all N samples over (N,H,W) per channel
      // batch is modified in-place.
      // xNormalized: [N] tensors storing the normalized values before gamma/beta scaling.
      // statsMean/statsVar: [N*C] per-sample stats (for InstanceNorm each sample's own stats;
      //                     for BatchNorm all N slices store the same batch-wide stats).
      //                     Indexed as n*C + c.
      static void propagate(std::vector<Tensor3D<T>*>& batch, const Shape3D& shape, NormParameters<T>& params,
                            const NormLayerConfig& config, LayerType normType, bool training,
                            std::vector<Tensor3D<T>>* xNormalized = nullptr, std::vector<T>* statsMean = nullptr,
                            std::vector<T>* statsVar = nullptr);

      //-- Backward pass --//

      // Unified backward pass for both InstanceNorm and BatchNorm.
      // dOutputs are modified in-place to contain dInput.
      // normType controls whether gradients use per-sample (H*W) or batch-wide (N*H*W) normalization.
      // statsMean/statsVar: [N*C] per-sample stats from the forward pass (indexed as n*C + c).
      static void backpropagate(std::vector<Tensor3D<T>*>& dOutputs, const Shape3D& shape,
                                const NormParameters<T>& params, const NormLayerConfig& config, LayerType normType,
                                const std::vector<T>& statsMean, const std::vector<T>& statsVar,
                                const std::vector<Tensor3D<T>>& xNormalized, std::vector<T>& dGamma,
                                std::vector<T>& dBeta);
  };
}

//===================================================================================================================//

#endif // CNN_NORMALIZATION_HPP
