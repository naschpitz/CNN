#ifndef CNN_BATCHNORM_HPP
#define CNN_BATCHNORM_HPP

#include "CNN_Types.hpp"
#include "CNN_BatchNormParameters.hpp"
#include "CNN_LayersConfig.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class BatchNorm
  {
    public:
      // Forward pass: always computes per-image spatial statistics (instance normalization).
      // If batchMean/batchVar/xNormalized pointers are provided, stores intermediates for
      // backpropagation and updates running stats. If pointers are null, just computes output.
      static Tensor3D<T> propagate(const Tensor3D<T>& input, const Shape3D& inputShape, BatchNormParameters<T>& params,
                                   const BatchNormLayerConfig& config, std::vector<T>* batchMean = nullptr,
                                   std::vector<T>* batchVar = nullptr, Tensor3D<T>* xNormalized = nullptr);

      // Backward pass: computes gradients for gamma, beta, and input
      static Tensor3D<T> backpropagate(const Tensor3D<T>& dOutput, const Shape3D& inputShape,
                                       const BatchNormParameters<T>& params, const BatchNormLayerConfig& config,
                                       const std::vector<T>& batchMean, const std::vector<T>& batchVar,
                                       const Tensor3D<T>& xNormalized, std::vector<T>& dGamma, std::vector<T>& dBeta);
  };
}

//===================================================================================================================//

#endif // CNN_BATCHNORM_HPP
