#ifndef CNN_TESTPROGRESS_HPP
#define CNN_TESTPROGRESS_HPP

#include <functional>
#include <sys/types.h>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  struct TestProgress {
      ulong currentSample;
      ulong totalSamples;
  };

  template <typename T>
  using TestCallback = std::function<void(const TestProgress<T>&)>;
}

//===================================================================================================================//

#endif // CNN_TESTPROGRESS_HPP
