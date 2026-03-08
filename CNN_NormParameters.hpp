#ifndef CNN_NORMPARAMETERS_HPP
#define CNN_NORMPARAMETERS_HPP

#include "CNN_Types.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN
{
  // Parameters shared by both BatchNorm and InstanceNorm layers:
  // gamma (scale) and beta (shift) are learnable, per-channel
  // runningMean and runningVar are accumulated during training, used at inference
  template <typename T>
  struct NormParameters {
      std::vector<T> gamma; // Scale: [numChannels]
      std::vector<T> beta; // Shift: [numChannels]
      std::vector<T> runningMean; // Running mean: [numChannels]
      std::vector<T> runningVar; // Running variance: [numChannels]
      ulong numChannels = 0;
  };
}

//===================================================================================================================//

#endif // CNN_NORMPARAMETERS_HPP
