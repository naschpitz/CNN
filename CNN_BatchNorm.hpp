#ifndef CNN_BATCHNORM_HPP
#define CNN_BATCHNORM_HPP

#include "CNN_Types.hpp"
#include "CNN_NormParameters.hpp"
#include "CNN_LayersConfig.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN
{
  // True Batch Normalization: normalizes per-channel across N×H×W (all samples in the mini-batch).
  // Single-sample inference is a particular case with N=1 (uses running stats when no intermediates requested).
  template <typename T>
  class BatchNorm
  {
    public:
      // Forward pass: takes a batch of inputs (N >= 1).
      // Training mode: provide batchMean/batchVar/xNormalized to compute batch-wide stats and store intermediates.
      // Inference mode: omit intermediates to use running stats instead.
      static std::vector<Tensor3D<T>> propagate(const std::vector<Tensor3D<T>>& inputs, const Shape3D& inputShape,
                                                NormParameters<T>& params, const NormLayerConfig& config,
                                                std::vector<T>* batchMean = nullptr, std::vector<T>* batchVar = nullptr,
                                                std::vector<Tensor3D<T>>* xNormalized = nullptr);

      // Backward pass: takes batch of upstream gradients, computes batch-aware
      // gradients for gamma, beta, and input across all samples.
      static std::vector<Tensor3D<T>> backpropagate(const std::vector<Tensor3D<T>>& dOutputs, const Shape3D& inputShape,
                                                    const NormParameters<T>& params, const NormLayerConfig& config,
                                                    const std::vector<T>& batchMean, const std::vector<T>& batchVar,
                                                    const std::vector<Tensor3D<T>>& xNormalized, std::vector<T>& dGamma,
                                                    std::vector<T>& dBeta);
  };
}

//===================================================================================================================//

#endif // CNN_BATCHNORM_HPP
