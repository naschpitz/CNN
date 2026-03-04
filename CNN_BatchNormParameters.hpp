#ifndef CNN_BATCHNORMPARAMETERS_HPP
#define CNN_BATCHNORMPARAMETERS_HPP

#include "CNN_Types.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN
{
  // Parameters for a single batch normalization layer:
  // gamma (scale) and beta (shift) are learnable, per-channel
  // runningMean and runningVar are accumulated during training, used at inference
  template <typename T>
  struct BatchNormParameters {
      std::vector<T> gamma; // Scale: [numChannels]
      std::vector<T> beta; // Shift: [numChannels]
      std::vector<T> runningMean; // Running mean: [numChannels]
      std::vector<T> runningVar; // Running variance: [numChannels]
      ulong numChannels = 0;
  };
}

//===================================================================================================================//

#endif // CNN_BATCHNORMPARAMETERS_HPP
