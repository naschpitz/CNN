#ifndef CNN_COREGPU_HPP
#define CNN_COREGPU_HPP

#include "CNN_Core.hpp"
#include "CNN_CoreGPUWorker.hpp"

#include <memory>
#include <vector>

//===================================================================================================================//

namespace CNN {
  template <typename T>
  class CoreGPU : public Core<T> {
    public:
      CoreGPU(const CoreConfig<T>& config);

      Output<T> predict(const Input<T>& input) override;
      void train(const Samples<T>& samples) override;
      TestResult<T> test(const Samples<T>& samples) override;

    private:
      //-- GPU workers (one per GPU) --//
      std::vector<std::unique_ptr<CoreGPUWorker<T>>> gpuWorkers;
      size_t numGPUs = 1;

      //-- Initialization --//
      void initializeWorkers();

      //-- Training coordination --//
      void mergeCNNGradients();
      void mergeANNParameters();
  };
}

//===================================================================================================================//

#endif // CNN_COREGPU_HPP

