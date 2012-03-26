#ifndef TRIANGLELIST_H
#define TRIANGLELIST_H

#include <vector>
#include <map>

class Triangle;
class Vector;

class TriangleList
{
public:
	TriangleList(void);
	~TriangleList(void);
	void AddTriangle(Triangle *tr, int compareTriangles);
	Triangle* GetTriangle(int index);
	int GetCount();
	std::vector<Triangle*> GetNeighbourTriangles(Vector *v);


protected:
	std::vector<Triangle*> triangles;
	std::map<Vector*,std::vector<Triangle*>> neighbourTriangles;

};
#endif
