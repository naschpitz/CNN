#ifndef CNN_TRAININGPROGRESS_HPP
#define CNN_TRAININGPROGRESS_HPP

#include <functional>
#include <sys/types.h>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  struct TrainingProgress {
    ulong currentEpoch;
    ulong totalEpochs;
    ulong currentSample;
    ulong totalSamples;
    T epochLoss;        // Average loss for completed epoch (0 if epoch not complete)
    T sampleLoss;       // Loss for current sample
  };

  template <typename T>
  using TrainingCallback = std::function<void(const TrainingProgress<T>&)>;
}

//===================================================================================================================//

#endif // CNN_TRAININGPROGRESS_HPP

