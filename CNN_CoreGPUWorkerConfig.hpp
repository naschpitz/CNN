#ifndef CNN_COREGPUWORKERCONFIG_HPP
#define CNN_COREGPUWORKERCONFIG_HPP

#include "CNN_CoreConfig.hpp"
#include "CNN_LogLevel.hpp"
#include "CNN_LayersConfig.hpp"
#include "CNN_CostFunctionConfig.hpp"
#include "CNN_TrainingConfig.hpp"
#include "CNN_Parameters.hpp"
#include "CNN_Types.hpp"

//===================================================================================================================//

namespace CNN
{
  // Configuration for a single GPU worker. Extracts only the fields needed
  // from CoreConfig and adds the per-GPU batch size.
  template <typename T>
  class CoreGPUWorkerConfig
  {
  public:
    //-- Constructor --//
    explicit CoreGPUWorkerConfig(const CoreConfig<T>& config);

    //-- CNN topology --//
    Shape3D inputShape;
    LayersConfig layersConfig;

    //-- Training --//
    TrainingConfig<T> trainingConfig;
    CostFunctionConfig<T> costFunctionConfig;

    //-- Parameters (initial weights) --//
    Parameters<T> parameters;

    //-- Per-GPU batch size --//
    ulong batchSize = 1;

    //-- Logging / progress --//
    ulong progressReports = 1000;
    LogLevel logLevel = LogLevel::ERROR;
  };
}

//===================================================================================================================//

#endif // CNN_COREGPUWORKERCONFIG_HPP
