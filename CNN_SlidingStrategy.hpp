#ifndef CNN_SLIDINGSTRATEGY_HPP
#define CNN_SLIDINGSTRATEGY_HPP

#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace CNN {
  enum class SlidingStrategyType {
    VALID,   // No padding
    FULL,    // Filter allowed outside input (pad = kernel - 1)
    SAME     // Output size equals input size when stride=1 (pad = floor(kernel/2))
  };

  const std::unordered_map<std::string, SlidingStrategyType> slidingStrategyMap = {
    {"valid", SlidingStrategyType::VALID},
    {"full", SlidingStrategyType::FULL},
    {"same", SlidingStrategyType::SAME},
  };

  class SlidingStrategy
  {
    public:
      static SlidingStrategyType nameToType(const std::string& name);
      static std::string typeToName(const SlidingStrategyType& type);

      // Compute padding for a given kernel size and strategy
      static ulong computePadding(ulong kernelSize, SlidingStrategyType strategy);
  };
}

//===================================================================================================================//

#endif // CNN_SLIDINGSTRATEGY_HPP

