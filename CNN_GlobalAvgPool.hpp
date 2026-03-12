#ifndef CNN_GLOBALAVGPOOL_HPP
#define CNN_GLOBALAVGPOOL_HPP

#include "CNN_Types.hpp"

namespace CNN
{

  template <typename T>
  class GlobalAvgPool
  {
    public:
      //-- Forward / Backward --//
      static void propagate(Tensor3D<T>& input, const Shape3D& inputShape);
      static void backpropagate(Tensor3D<T>& gradOutput, const Shape3D& inputShape);
  };

} // namespace CNN

#endif // CNN_GLOBALAVGPOOL_HPP
