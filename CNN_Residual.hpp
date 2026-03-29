#ifndef CNN_RESIDUAL_HPP
#define CNN_RESIDUAL_HPP

#include "CNN_Parameters.hpp"
#include "CNN_Tensor3D.hpp"

namespace CNN
{
  template <typename T>
  class Residual
  {
    public:
      // Forward: add skip connection to block output.
      // If projection is nullptr, identity shortcut (same channels).
      // If projection is provided, applies 1×1 conv to skip before adding.
      static void propagate(Tensor3D<T>& blockOutput, const Tensor3D<T>& skipInput,
                            const ResidualParameters<T>* projection);

      // Backward: split gradient to skip path, compute projection gradients if needed.
      // dBlockOutput is modified in-place (gradient continues backward through block).
      // Returns dSkip — the gradient for the skip input.
      // If projection exists, also accumulates dWeights and dBiases.
      static Tensor3D<T> backpropagate(const Tensor3D<T>& dBlockOutput, const Tensor3D<T>& skipInput,
                                       const ResidualParameters<T>* projection,
                                       ResidualParameters<T>* dProjection);
  };
}

#endif // CNN_RESIDUAL_HPP

