#include "CNN_Shape3D.hpp"

using namespace CNN;

//===================================================================================================================//

ulong Shape3D::size() const
{
  return c * h * w;
}

//===================================================================================================================//

bool Shape3D::operator==(const Shape3D& other) const
{
  return c == other.c && h == other.h && w == other.w;
}

//===================================================================================================================//

bool Shape3D::operator!=(const Shape3D& other) const
{
  return !(*this == other);
}
