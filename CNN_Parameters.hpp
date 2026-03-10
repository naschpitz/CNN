#ifndef CNN_PARAMETERS_HPP
#define CNN_PARAMETERS_HPP

#include "CNN_Conv2DParameters.hpp"
#include "CNN_InstanceNormParameters.hpp"

#include <ANN_Parameters.hpp>

#include <vector>

//===================================================================================================================//

namespace CNN
{
  // All CNN parameters (conv layers + batch norm layers + ANN dense parameters)
  template <typename T>
  struct Parameters {
      std::vector<ConvParameters<T>> convParams; // One per conv layer
      std::vector<InstanceNormParameters<T>> inParams; // One per batch norm layer
      ANN::Parameters<T> denseParams; // Dense layer parameters (delegated to ANN)
  };
}

//===================================================================================================================//

#endif // CNN_PARAMETERS_HPP
