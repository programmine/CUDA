#include "trianglelist.h"
#include "triangle.h"
#include <iostream> 
#include <algorithm>

TriangleList::TriangleList(void)
{
	triangles.clear();
}

TriangleList::~TriangleList(void)
{
	for (int index=0;index<triangles.size();index++)
	{
		delete triangles.at(index);
	}
	triangles.clear();
}


void TriangleList::AddTriangle(Triangle *triangle, int compareTriangles)
{
	if (triangles.size() < compareTriangles) compareTriangles=triangles.size();
	for (int index=triangles.size()-compareTriangles; index<triangles.size();index++)
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

Triangle* TriangleList::GetTriangle(int index)
{
	if ((index < 0) && (index > triangles.size())) return NULL;
	return triangles.at(index);
}

int TriangleList::GetCount()
{
	return triangles.size();
}


std::vector<Triangle*> TriangleList::GetNeighbourTriangles(Vector *v)
{
	return neighbourTriangles[v];
}