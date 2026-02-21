#ifndef CNN_TRAININGMETADATA_HPP
#define CNN_TRAININGMETADATA_HPP

#include <string>
#include <sys/types.h>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  struct TrainingMetadata {
    std::string startTime;         // ISO 8601 format
    std::string endTime;           // ISO 8601 format
    double durationSeconds;        // Total training duration in seconds
    std::string durationFormatted; // Human-readable duration
    ulong numSamples;              // Number of training samples used
    T finalLoss;                   // Average loss at end of training
  };
}

//===================================================================================================================//

#endif // CNN_TRAININGMETADATA_HPP

