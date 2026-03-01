#ifndef CNN_CORECPU_HPP
#define CNN_CORECPU_HPP

#include "CNN_Core.hpp"
#include "CNN_CoreCPUWorker.hpp"

#include <memory>
#include <vector>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  class CoreCPU : public Core<T> {
    public:
      //-- Constructor --//
      CoreCPU(const CoreConfig<T>& config);

      //-- Core interface --//
      Output<T> predict(const Input<T>& input) override;
      void train(ulong numSamples, const SampleProvider<T>& sampleProvider) override;
      TestResult<T> test(const Samples<T>& samples) override;

    private:
      //-- Step-by-step worker (for predict / single-threaded path) --//
      std::unique_ptr<CoreCPUWorker<T>> stepWorker;

      //-- Global CNN gradient accumulators (for merging worker results) --//
      std::vector<std::vector<T>> accumDConvFilters;
      std::vector<std::vector<T>> accumDConvBiases;

      //-- Training helpers --//
      void resetGlobalCNNAccumulators();
      void mergeWorkerCNNAccumulators(const CoreCPUWorker<T>& worker);
      void updateCNNParameters(ulong numSamples);
  };
}

//===================================================================================================================//

#endif // CNN_CORECPU_HPP

