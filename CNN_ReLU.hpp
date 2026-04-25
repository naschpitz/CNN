#ifndef CNN_RELU_HPP
#define CNN_RELU_HPP

#include "CNN_Types.hpp"

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class ReLU
  {
  public:
    // Propagate: element-wise max(0, x)
    static Tensor3D<T> propagate(const Tensor3D<T>& input);

    // Backpropagation: gradient passes through where input > 0
    // dOut: gradient of loss w.r.t. output of this layer
    // input: the input that was passed to propagate()
    // Returns: gradient of loss w.r.t. input
    static Tensor3D<T> backpropagate(const Tensor3D<T>& dOut, const Tensor3D<T>& input);
  };
}

//===================================================================================================================//

#endif // CNN_RELU_HPP
