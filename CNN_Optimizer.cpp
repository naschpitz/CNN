#include "CNN_Optimizer.hpp"

//===================================================================================================================//

namespace CNN
{

  template <typename T>
  OptimizerType Optimizer<T>::nameToType(const std::string& name)
  {
    if (name == "sgd")
      return OptimizerType::SGD;

    if (name == "adam")
      return OptimizerType::ADAM;

    throw std::runtime_error("Unknown optimizer type: " + name);
  }

  //===================================================================================================================//

  template <typename T>
  std::string Optimizer<T>::typeToName(OptimizerType t)
  {
    switch (t) {
    case OptimizerType::SGD:
      return "sgd";
    case OptimizerType::ADAM:
      return "adam";
    default:
      return "sgd";
    }
  }

  //===================================================================================================================//
  // Explicit template instantiations
  //===================================================================================================================//

  template struct Optimizer<int>;
  template struct Optimizer<float>;
  template struct Optimizer<double>;

} // namespace CNN