#ifndef CNN_INSTANCENORM_HPP
#define CNN_INSTANCENORM_HPP

#include "CNN_Types.hpp"
#include "CNN_InstanceNormParameters.hpp"
#include "CNN_LayersConfig.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class InstanceNorm
  {
    public:
      // Forward pass: when training=false uses running mean/var; when training=true computes
      // batch statistics, stores normalized values for backpropagation, and updates running stats.
      static Tensor3D<T> propagate(const Tensor3D<T>& input, const Shape3D& inputShape,
                                   InstanceNormParameters<T>& params, const InstanceNormLayerConfig& config,
                                   bool training = false, std::vector<T>* batchMean = nullptr,
                                   std::vector<T>* batchVar = nullptr, Tensor3D<T>* xNormalized = nullptr);

      // Backward pass: computes gradients for gamma, beta, and input
      static Tensor3D<T> backpropagate(const Tensor3D<T>& dOutput, const Shape3D& inputShape,
                                       const InstanceNormParameters<T>& params, const InstanceNormLayerConfig& config,
                                       const std::vector<T>& batchMean, const std::vector<T>& batchVar,
                                       const Tensor3D<T>& xNormalized, std::vector<T>& dGamma, std::vector<T>& dBeta);
  };
}

//===================================================================================================================//

#endif // CNN_INSTANCENORM_HPP
