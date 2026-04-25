#ifndef CNN_RESIDUALPARAMETERS_HPP
#define CNN_RESIDUALPARAMETERS_HPP

#include <vector>

//===================================================================================================================//

namespace CNN
{
  // 1×1 convolution projection for residual skip connections with channel mismatch
  template <typename T>
  struct ResidualParameters {
    std::vector<T> weights; // outC × inC (1×1 conv, no spatial kernel)
    std::vector<T> biases; // outC
    ulong inC = 0;
    ulong outC = 0;
  };
}

//===================================================================================================================//

#endif // CNN_RESIDUALPARAMETERS_HPP
