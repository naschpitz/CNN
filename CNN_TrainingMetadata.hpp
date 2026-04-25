#ifndef CNN_TRAININGMETADATA_HPP
#define CNN_TRAININGMETADATA_HPP

#include <string>
#include <sys/types.h>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  struct TrainingMetadata {
    std::string startTime; // ISO 8601 format
    std::string endTime; // ISO 8601 format
    double durationSeconds; // Total training duration in seconds
    std::string durationFormatted; // Human-readable duration
    ulong numSamples; // Number of training samples used
    T finalLoss; // Average loss at end of training

    // Monitoring fields
    ulong lastEpoch = 0; // Epoch at which this model was saved
    std::string stopReason; // Why training stopped (empty = completed all epochs)
    ulong bestEpoch = 0; // Epoch with best loss
    T bestLoss = 0; // Best loss value
  };
}

//===================================================================================================================//

#endif // CNN_TRAININGMETADATA_HPP
