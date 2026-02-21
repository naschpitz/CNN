#ifndef CNN_TRAININGCONFIG_HPP
#define CNN_TRAININGCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  struct TrainingConfig {
    ulong numEpochs = 0;
    float learningRate = 0.01f;
    int numThreads = 0;           // 0 = use all available cores (for CPU mode)
    ulong progressReports = 1000; // Number of progress reports per epoch (0 = no reports, default = 1000)
  };
}

//===================================================================================================================//

#endif // CNN_TRAININGCONFIG_HPP

