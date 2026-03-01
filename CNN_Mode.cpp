#include "CNN_Mode.hpp"

#include <stdexcept>

using namespace CNN;

//===================================================================================================================//

ModeType Mode::nameToType(const std::string& name)
{
  auto it = modeMap.find(name);

  if (it != modeMap.end()) {
    return it->second;
  }

  throw std::runtime_error("Unknown mode type: " + name);
}

//===================================================================================================================//

std::string Mode::typeToName(const ModeType& modeType)
{
  for (const auto& pair : modeMap) {
    if (pair.second == modeType) {
      return pair.first;
    }
  }

  throw std::runtime_error("Unknown mode type enum value");
}
