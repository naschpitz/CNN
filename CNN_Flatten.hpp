#ifndef CNN_FLATTEN_HPP
#define CNN_FLATTEN_HPP

#include "CNN_Types.hpp"

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class Flatten
  {
    public:
      // Propagate: reshape 3D tensor to 1D vector
      static Tensor1D<T> propagate(const Tensor3D<T>& input);

      // Backpropagation: reshape 1D gradient back to 3D tensor
      // dOut: gradient of loss w.r.t. flattened output (1D)
      // inputShape: the shape of the original 3D input
      // Returns: gradient reshaped to 3D
      static Tensor3D<T> backpropagate(const Tensor1D<T>& dOut, const Shape3D& inputShape);
  };
}

//===================================================================================================================//

#endif // CNN_FLATTEN_HPP
