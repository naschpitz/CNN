#ifndef CNN_WORKER_HPP
#define CNN_WORKER_HPP

#include "CNN_CostFunctionConfig.hpp"
#include "CNN_LayersConfig.hpp"
#include "CNN_Parameters.hpp"
#include "CNN_Types.hpp"

//===================================================================================================================//

namespace CNN
{
  template <typename T>
  class Worker
  {
    public:
      virtual ~Worker() = default;

      //-- Parameter initialization (shared by CPU and GPU workers) --//
      static void initializeConvParams(const LayersConfig& layersConfig, const Shape3D& inputShape,
                                       Parameters<T>& parameters, ulong seed);
      static void initializeNormParams(const LayersConfig& layersConfig, const Shape3D& inputShape,
                                       Parameters<T>& parameters);

      //-- Loss calculation (shared by CPU and GPU workers) --//
      T calculateLoss(const Output<T>& predicted, const Output<T>& expected) const;

    protected:
      CostFunctionConfig<T> costFunctionConfig;
  };
}

//===================================================================================================================//

#endif // CNN_WORKER_HPP
