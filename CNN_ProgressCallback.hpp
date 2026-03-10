#ifndef CNN_PROGRESSCALLBACK_HPP
#define CNN_PROGRESSCALLBACK_HPP

#include <functional>
#include <sys/types.h>

//===================================================================================================================//

namespace CNN
{
  // Generic progress callback for test and predict modes.
  // Parameters: (currentSample, totalSamples)
  using ProgressCallback = std::function<void(ulong, ulong)>;
}

//===================================================================================================================//

#endif // CNN_PROGRESSCALLBACK_HPP
