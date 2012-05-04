#ifndef TRIANGLELISTCUDA_H
#define TRIANGLELISTCUDA_H

#include <vector>
#include <map>

class Triangle;
class Vector;
#include "trianglelist.h"

class TriangleListCUDA : public TriangleList
{
public:
	void AddTriangle(Triangle *tr, int compareTriangles);

};
#endif