#ifndef CNN_MODE_HPP
#define CNN_MODE_HPP

#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace CNN
{
  enum class ModeType { TRAIN, PREDICT, TEST };

  const std::unordered_map<std::string, ModeType> modeMap = {
    {"train", ModeType::TRAIN},
    {"predict", ModeType::PREDICT},
    {"test", ModeType::TEST},
  };

  class Mode
  {
  public:
    static ModeType nameToType(const std::string& name);
    static std::string typeToName(const ModeType& modeType);
  };
}

//===================================================================================================================//

#endif // CNN_MODE_HPP
