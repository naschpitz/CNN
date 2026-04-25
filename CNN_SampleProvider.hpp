#ifndef CNN_SAMPLEPROVIDER_HPP
#define CNN_SAMPLEPROVIDER_HPP

#include "CNN_Sample.hpp"

#include <algorithm>
#include <functional>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  // Lazy supplier used by train() and test(): given the full shuffled index
  // array, a batch size, and a 0-based batch index, returns the corresponding
  // batch of samples. Lets callers stream samples on demand (e.g. decode
  // images batch-by-batch) instead of holding the full Samples<T> vector
  // in memory.
  template <typename T>
  using SampleProvider =
    std::function<Samples<T>(const std::vector<ulong>& sampleIndices, ulong batchSize, ulong batchIndex)>;

  // Build a SampleProvider that serves from an in-memory Samples vector.
  // No prefetching needed — all data is already in memory.
  template <typename T>
  inline SampleProvider<T> makeSampleProvider(const Samples<T>& samples)
  {
    return [&samples](const std::vector<ulong>& sampleIndices, ulong batchSize, ulong batchIndex) -> Samples<T> {
      ulong start = batchIndex * batchSize;
      ulong end = std::min(start + batchSize, static_cast<ulong>(sampleIndices.size()));

      Samples<T> batch;
      batch.reserve(end - start);

      for (ulong i = start; i < end; i++) {
        batch.push_back(samples[sampleIndices[i]]);
      }

      return batch;
    };
  }
}

//===================================================================================================================//

#endif // CNN_SAMPLEPROVIDER_HPP
