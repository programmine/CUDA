#ifndef TRIANGLELISTCUDA_H
#define TRIANGLELISTCUDA_H

#include <vector>
#include <map>
#include <cutil_math.h>

class Triangle;
class Vector;
#include "trianglelist.h"

class TriangleListCUDA : public TriangleList
{
public:
	void AddTriangle(Triangle *tr, unsigned int compareTriangles);
	void ToCUDADataStructure(int *triangles,int *neighbours,float3 *v1,float3 *v2,float3 *v3);

};
#endif