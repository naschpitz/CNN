#ifndef CNN_SHAPE3D_HPP
#define CNN_SHAPE3D_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace CNN
{
  // Shape of a 3D tensor (Channels, Height, Width) - NCHW layout
  class Shape3D
  {
  public:
    ulong c; // channels
    ulong h; // height
    ulong w; // width

    //-- Methods --//
    ulong size() const;
    bool operator==(const Shape3D& other) const;
    bool operator!=(const Shape3D& other) const;
  };
}

//===================================================================================================================//

#endif // CNN_SHAPE3D_HPP
