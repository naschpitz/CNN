#ifndef CNN_CORECONFIG_HPP
#define CNN_CORECONFIG_HPP

#include "CNN_Mode.hpp"
#include "CNN_Device.hpp"
#include "CNN_LogLevel.hpp"
#include "CNN_LayersConfig.hpp"
#include "CNN_CostFunctionConfig.hpp"
#include "CNN_TrainingConfig.hpp"
#include "CNN_Parameters.hpp"
#include "CNN_Types.hpp"

//===================================================================================================================//

namespace CNN {
  template <typename T>
  struct CoreConfig {
    ModeType modeType;
    DeviceType deviceType;
    Shape3D inputShape;            // Input tensor shape (C, H, W)
    LayersConfig layersConfig;
    CostFunctionConfig<T> costFunctionConfig;
    TrainingConfig<T> trainingConfig;
    Parameters<T> parameters;
    ulong progressReports = 1000;  // Number of progress reports (0 = no reports, default = 1000)
    LogLevel logLevel = LogLevel::ERROR;
  };
}

//===================================================================================================================//

#endif // CNN_CORECONFIG_HPP

