#ifndef CNN_BATCHNORM_HPP
#define CNN_BATCHNORM_HPP

#include "CNN_Types.hpp"
#include "CNN_NormParameters.hpp"
#include "CNN_LayersConfig.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class BatchNorm
  {
    public:
      // Forward pass: computes batch-wide statistics across all N samples.
      // When training=true, computes mean/var across (N, H, W) per channel,
      // normalizes all samples, and stores intermediates for backward.
      // When training=false, uses running mean/var for inference.
      static void propagate(std::vector<Tensor3D<T>*>& batch, const Shape3D& shape, NormParameters<T>& params,
                            const NormLayerConfig& config, bool training,
                            std::vector<Tensor3D<T>>* xNormalized = nullptr, std::vector<T>* batchMean = nullptr,
                            std::vector<T>* batchVar = nullptr);

      // Backward pass: computes dGamma, dBeta, and dInput for all samples.
      // dOutputs are modified in-place to contain dInput.
      static void backpropagate(std::vector<Tensor3D<T>*>& dOutputs, const Shape3D& shape,
                                const NormParameters<T>& params, const NormLayerConfig& config,
                                const std::vector<T>& batchMean, const std::vector<T>& batchVar,
                                const std::vector<Tensor3D<T>>& xNormalized, std::vector<T>& dGamma,
                                std::vector<T>& dBeta);
  };
}

//===================================================================================================================//

#endif // CNN_BATCHNORM_HPP
