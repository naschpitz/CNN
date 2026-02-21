#ifndef CNN_SAMPLE_HPP
#define CNN_SAMPLE_HPP

#include "CNN_Types.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  struct Sample {
    Input<T> input;    // 3D tensor (e.g., 1x28x28 for MNIST)
    Output<T> output;  // 1D expected output (e.g., [0,0,0,1,0,0,0,0,0,0] for digit 3)
  };

  template <typename T>
  using Samples = std::vector<Sample<T>>;
}

//===================================================================================================================//

#endif // CNN_SAMPLE_HPP

