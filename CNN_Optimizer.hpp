#ifndef CNN_OPTIMIZER_HPP
#define CNN_OPTIMIZER_HPP

#include <stdexcept>
#include <string>
#include <sys/types.h>

//===================================================================================================================//

namespace CNN
{
  enum class OptimizerType : int { SGD = 0, ADAM = 1 };

  template <typename T>
  struct Optimizer {
    //-- Configuration --//
    OptimizerType type = OptimizerType::SGD;
    T beta1 = static_cast<T>(0.9);
    T beta2 = static_cast<T>(0.999);
    T epsilon = static_cast<T>(1e-8);

    //-- Name/type conversion --//
    static OptimizerType nameToType(const std::string& name);
    static std::string typeToName(OptimizerType t);
  };
}

//===================================================================================================================//

#endif // CNN_OPTIMIZER_HPP
