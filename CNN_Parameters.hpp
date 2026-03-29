#ifndef CNN_PARAMETERS_HPP
#define CNN_PARAMETERS_HPP

#include "CNN_Conv2DParameters.hpp"
#include "CNN_NormParameters.hpp"

#include <ANN_Parameters.hpp>

#include <vector>

//===================================================================================================================//

namespace CNN
{
  // 1×1 convolution projection for residual skip connections with channel mismatch
  template <typename T>
  struct ResidualParameters {
      std::vector<T> weights; // outC × inC (1×1 conv, no spatial kernel)
      std::vector<T> biases;  // outC
      ulong inC = 0;
      ulong outC = 0;
  };

  // All CNN parameters (conv layers + batch norm layers + residual projections + ANN dense parameters)
  template <typename T>
  struct Parameters {
      std::vector<ConvParameters<T>> convParams;         // One per conv layer
      std::vector<NormParameters<T>> normParams;         // One per norm layer
      std::vector<ResidualParameters<T>> residualParams;  // One per residual block with channel mismatch
      ANN::Parameters<T> denseParams;                     // Dense layer parameters (delegated to ANN)
  };
}

//===================================================================================================================//

#endif // CNN_PARAMETERS_HPP
