#ifndef CNN_INSTANCENORMPARAMETERS_HPP
#define CNN_INSTANCENORMPARAMETERS_HPP

#include "CNN_Types.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN
{
  // Parameters for a single batch normalization layer:
  // gamma (scale) and beta (shift) are learnable, per-channel
  // runningMean and runningVar are accumulated during training, used at inference
  template <typename T>
  struct InstanceNormParameters {
      std::vector<T> gamma; // Scale: [numChannels]
      std::vector<T> beta; // Shift: [numChannels]
      std::vector<T> runningMean; // Running mean: [numChannels]
      std::vector<T> runningVar; // Running variance: [numChannels]
      ulong numChannels = 0;
  };
}

//===================================================================================================================//

#endif // CNN_INSTANCENORMPARAMETERS_HPP
