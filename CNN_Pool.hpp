#ifndef CNN_POOL_HPP
#define CNN_POOL_HPP

#include "CNN_Types.hpp"
#include "CNN_LayersConfig.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class Pool
  {
    public:
      // Propagate: apply max or avg pooling
      // maxIndices: [out] for max pooling, stores index of max element per output position
      //             (needed for backpropagation). Empty for avg pooling.
      static Tensor3D<T> propagate(const Tensor3D<T>& input, const PoolLayerConfig& config,
                                   std::vector<ulong>& maxIndices);

      // Backpropagation: distribute gradient back through pooling
      // dOut: gradient of loss w.r.t. output of this layer [C x outH x outW]
      // inputShape: shape of the input that was passed to propagate()
      // config: pooling layer config
      // maxIndices: the indices saved during propagation (for max pooling)
      // Returns: gradient of loss w.r.t. input [C x inputH x inputW]
      static Tensor3D<T> backpropagate(const Tensor3D<T>& dOut, const Shape3D& inputShape,
                                       const PoolLayerConfig& config, const std::vector<ulong>& maxIndices);
  };
}

//===================================================================================================================//

#endif // CNN_POOL_HPP
