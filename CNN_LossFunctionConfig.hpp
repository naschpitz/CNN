#ifndef CNN_LOSSFUNCTIONCONFIG_HPP
#define CNN_LOSSFUNCTIONCONFIG_HPP

#include <string>
#include <stdexcept>
#include <vector>

//===================================================================================================================//

namespace CNN {
  enum class LossFunctionType : int {
    SQUARED_DIFFERENCE = 0,
    WEIGHTED_SQUARED_DIFFERENCE = 1
  };

  //-- String conversion helpers --//
  struct LossFunction {
    static LossFunctionType nameToType(const std::string& name) {
      if (name == "squaredDifference") return LossFunctionType::SQUARED_DIFFERENCE;
      if (name == "weightedSquaredDifference") return LossFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
      throw std::runtime_error("Unknown loss function type: " + name);
    }

    static std::string typeToName(LossFunctionType type) {
      switch (type) {
        case LossFunctionType::SQUARED_DIFFERENCE: return "squaredDifference";
        case LossFunctionType::WEIGHTED_SQUARED_DIFFERENCE: return "weightedSquaredDifference";
        default: return "squaredDifference";
      }
    }
  };

  template <typename T>
  struct LossFunctionConfig {
    LossFunctionType type = LossFunctionType::SQUARED_DIFFERENCE;
    std::vector<T> weights;  // Per-output-neuron weights (only used for WEIGHTED_SQUARED_DIFFERENCE)
  };
}

//===================================================================================================================//

#endif // CNN_LOSSFUNCTIONCONFIG_HPP

