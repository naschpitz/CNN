#ifndef CNN_COREGPU_HPP
#define CNN_COREGPU_HPP

#include "CNN_Core.hpp"
#include "CNN_CoreGPUWorker.hpp"

#include <memory>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class CoreGPU : public Core<T>
  {
    public:
      CoreGPU(const CoreConfig<T>& config);

      using Core<T>::predict; // Bring in eager Inputs<T> + single-input convenience wrappers
      PredictResults<T> predict(ulong numSamples, const InputProvider<T>& provider) override;
      void train(ulong numSamples, const SampleProvider<T>& sampleProvider) override;
      TestResult<T> test(ulong numSamples, const SampleProvider<T>& sampleProvider) override;
      void syncParametersToGPU() override;

      //-- Worker access (for diagnostics/testing) --//
      CoreGPUWorker<T>* getWorker(size_t idx = 0)
      {
        return (idx < gpuWorkers.size()) ? gpuWorkers[idx].get() : nullptr;
      }

    private:
      //-- GPU workers (one per GPU) --//
      std::vector<std::unique_ptr<CoreGPUWorker<T>>> gpuWorkers;
      size_t numGPUs = 1;

      //-- Initialization --//
      void initializeWorkers();

      //-- Training coordination --//
      void mergeCNNGradients();
      void mergeANNGradients();
  };
}

//===================================================================================================================//

#endif // CNN_COREGPU_HPP
