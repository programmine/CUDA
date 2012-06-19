#include "trianglelistCUDA.h"
#include "triangle.h"
#include <iostream> 
#include <algorithm>


void TriangleListCUDA::AddTriangle(Triangle *triangle, unsigned int compareTriangles)
{
	if (triangles.size() < compareTriangles) compareTriangles=triangles.size();
	for (unsigned int index=triangles.size()-compareTriangles; index<triangles.size();index++)
	{
		Triangle *tr = triangles.at(index);
		if (tr->IsAdjacentVector(triangle->Point1))
		{
			std::vector<Triangle*>::iterator it = (std::find(neighbourTriangles[triangle->Point1].begin(),neighbourTriangles[triangle->Point1].end(),tr));
			if (neighbourTriangles[triangle->Point1].end()==it) neighbourTriangles[triangle->Point1].push_back(tr);
		}
		if (tr->IsAdjacentVector(triangle->Point2))
		{
			std::vector<Triangle*>::iterator it = (std::find(neighbourTriangles[triangle->Point2].begin(),neighbourTriangles[triangle->Point2].end(),tr));
			if (neighbourTriangles[triangle->Point2].end()==it) neighbourTriangles[triangle->Point2].push_back(tr);
		}
		if (tr->IsAdjacentVector(triangle->Point3))
		{
			std::vector<Triangle*>::iterator it = (std::find(neighbourTriangles[triangle->Point3].begin(),neighbourTriangles[triangle->Point3].end(),tr));
			if (neighbourTriangles[triangle->Point3].end()==it) neighbourTriangles[triangle->Point3].push_back(tr);
		}

	}
	triangles.push_back(triangle);
}


void TriangleListCUDA::ToCUDADataStructure(int *trianglesCUDA,int *neighboursCUDA,float3 *v1CUDA,float3 *v2CUDA,float3 *v3CUDA)
{
	trianglesCUDA = new int[triangles.size()]();
	neighboursCUDA = new int[neighbourTriangles.size()]();
}