#include "triangle.h"
#include <iostream>

Triangle::Triangle(void)
{
	this->Point1 = new Vector(0,0,0);
	this->Point2 = new Vector(0,0,0);
	this->Point3 = new Vector(0,0,0);
}

Triangle::Triangle(Vector *p1,Vector *p2,Vector *p3)
{
	this->Point1 = p1;
	this->Point2 = p2;
	this->Point3 = p3;
}

Triangle::~Triangle(void)
{
}

Vector* Triangle::UpdateNormal()
{
	Vector v1 = *Point1 - *Point2;
	Vector v2 = *Point3 - *Point2;
	this->Normal = &(v2.crossProduct(v1)); 
	return this->Normal;
}


bool Triangle::IsAdjacentVector(Vector *v)
{
	if (*(this->Point1) == *v) return true;
	if (*(this->Point2) == *v) return true;
	if (*(this->Point3) == *v) return true;

	return false;
}