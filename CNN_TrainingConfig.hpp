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
  };
}

//===================================================================================================================//

#endif // CNN_TRAININGCONFIG_HPP

