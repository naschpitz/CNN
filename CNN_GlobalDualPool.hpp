#ifndef CNN_GLOBALDUALPOOL_HPP
#define CNN_GLOBALDUALPOOL_HPP

#include "CNN_LayersConfig.hpp"
#include "CNN_Tensor3D.hpp"

namespace CNN
{
  template <typename T>
  class GlobalDualPool
  {
  public:
    // Forward: (C, H, W) -> (2C, 1, 1), first C = avg, last C = max
    static void propagate(Tensor3D<T>& input, const Shape3D& inputShape);

    // Backward: distributes gradient from (2C, 1, 1) back to (C, H, W)
    // avg gradient: uniform 1/spatialSize; max gradient: 1.0 at max index only
    static void backpropagate(Tensor3D<T>& gradOutput, const Tensor3D<T>& layerInput, const Shape3D& inputShape);
  };
}

#endif // CNN_GLOBALDUALPOOL_HPP
