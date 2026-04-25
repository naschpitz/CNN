#ifndef CNN_POOLTYPE_HPP
#define CNN_POOLTYPE_HPP

#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace CNN
{
  enum class PoolTypeEnum { MAX, AVG };

  const std::unordered_map<std::string, PoolTypeEnum> poolTypeMap = {
    {"max", PoolTypeEnum::MAX},
    {"avg", PoolTypeEnum::AVG},
  };

  class PoolType
  {
  public:
    static PoolTypeEnum nameToType(const std::string& name);
    static std::string typeToName(const PoolTypeEnum& type);
  };
}

//===================================================================================================================//

#endif // CNN_POOLTYPE_HPP
