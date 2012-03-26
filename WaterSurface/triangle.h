#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "vector.h"

class Triangle
{
public:
	Triangle(void);
	Triangle(Vector *p1,Vector *p2,Vector *p3);
	~Triangle(void);
	Vector* UpdateNormal();
	Vector* Point1;
	Vector* Point2;
	Vector* Point3;
	Vector* Normal;
	bool IsAdjacentVector(Vector *v);
};
#endif