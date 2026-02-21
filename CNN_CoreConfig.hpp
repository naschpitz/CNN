#ifndef CNN_CORECONFIG_HPP
#define CNN_CORECONFIG_HPP

#include "CNN_Mode.hpp"
#include "CNN_Device.hpp"
#include "CNN_LayersConfig.hpp"
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
    TrainingConfig<T> trainingConfig;
    Parameters<T> parameters;
    bool verbose = false;
  };
}

//===================================================================================================================//

#endif // CNN_CORECONFIG_HPP

