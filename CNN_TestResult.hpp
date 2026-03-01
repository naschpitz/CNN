#ifndef CNN_TESTRESULT_HPP
#define CNN_TESTRESULT_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  struct TestResult {
      ulong numSamples; // Total number of samples tested
      T totalLoss; // Sum of all sample losses
      T averageLoss; // Average loss per sample (totalLoss / numSamples)
      ulong numCorrect; // Number of correctly classified samples (argmax match)
      T accuracy; // Percentage of correct classifications (0-100)
  };
}

//===================================================================================================================//

#endif // CNN_TESTRESULT_HPP
