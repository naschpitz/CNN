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
    int numThreads = 0;           // 0 = use all available cores (for CPU mode)
    int numGPUs = 0;              // 0 = use all available GPUs (for GPU mode)
  };
}

//===================================================================================================================//

#endif // CNN_TRAININGCONFIG_HPP

