#ifndef CNN_TESTCONFIG_HPP
#define CNN_TESTCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace CNN
{
  struct TestConfig {
      ulong batchSize = 64; // Mini-batch size for test evaluation (default = 64)
  };
}

//===================================================================================================================//

#endif // CNN_TESTCONFIG_HPP
