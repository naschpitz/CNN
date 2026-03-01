#ifndef CNN_PREDICTMETADATA_HPP
#define CNN_PREDICTMETADATA_HPP

#include <string>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  struct PredictMetadata {
      std::string startTime; // ISO 8601 format
      std::string endTime; // ISO 8601 format
      double durationSeconds; // Total predict duration in seconds
      std::string durationFormatted; // Human-readable duration
  };
}

//===================================================================================================================//

#endif // CNN_PREDICTMETADATA_HPP
