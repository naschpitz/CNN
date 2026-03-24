#ifndef CNN_TYPES_HPP
#define CNN_TYPES_HPP

#include "CNN_Shape3D.hpp"
#include "CNN_Tensor3D.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN
{
  // 1D tensor (used for flattened output)
  template <typename T>
  using Tensor1D = std::vector<T>;

  // Input to the CNN is a 3D tensor
  template <typename T>
  using Input = Tensor3D<T>;

  // Output from the CNN is a 1D vector (from dense layers)
  template <typename T>
  using Output = std::vector<T>;

  // Batch types
  template <typename T>
  using Inputs = std::vector<Input<T>>;

  template <typename T>
  using Outputs = std::vector<Output<T>>;
}

//===================================================================================================================//

#endif // CNN_TYPES_HPP
