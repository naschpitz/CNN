#include "CNN_CoreGPUWorkerConfig.hpp"

//===================================================================================================================//

namespace CNN
{

  //===================================================================================================================//

  template <typename T>
  CoreGPUWorkerConfig<T>::CoreGPUWorkerConfig(const CoreConfig<T>& config)
    : inputShape(config.inputShape),
      layersConfig(config.layersConfig),
      trainingConfig(config.trainingConfig),
      costFunctionConfig(config.costFunctionConfig),
      parameters(config.parameters),
      batchSize(1),
      progressReports(config.progressReports),
      logLevel(config.logLevel)
  {
  }

  //===================================================================================================================//
  // Template instantiations
  //===================================================================================================================//

  template class CoreGPUWorkerConfig<int>;
  template class CoreGPUWorkerConfig<float>;
  template class CoreGPUWorkerConfig<double>;

}
