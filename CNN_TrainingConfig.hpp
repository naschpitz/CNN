#ifndef CNN_TRAININGCONFIG_HPP
#define CNN_TRAININGCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  struct TrainingConfig {
    ulong numEpochs = 0;
    float learningRate = 0.01f;
    ulong batchSize = 64;         // Mini-batch size (default = 64)
    bool shuffleSamples = true;   // Shuffle sample order each epoch (default = true)
    float dropoutRate = 0.0f;     // Dropout probability for dense hidden layers (0.0 = disabled)
  };
}

//===================================================================================================================//

#endif // CNN_TRAININGCONFIG_HPP

