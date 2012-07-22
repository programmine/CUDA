#include "vector.h"
#include <iostream> 
#include <math.h>


Vector::Vector()
{
	x = 0;
	y = 0;
	z = 0;
}

Vector::Vector(float _x,float _y,float _z)
{
	x = _x;
	y = _y;
	z = _z;
}

Vector::~Vector(void)
{
}



Vector Vector::crossProduct(Vector other)
{
	Vector cross(0.0f,0.0f,0.0f);
	cross.x = y * other.z - other.y * z;
	cross.y = z * other.x - other.z * x;
	cross.z = x * other.y - other.x * y;
	return cross;
}


Vector Vector::operator+(Vector other){
	Vector v(other.x + x,other.y+y,other.z+z);
	return v;
}


Vector Vector::operator-(Vector other){
	Vector v(-other.x+x,-other.y+y,-other.z+z);
	return v;
}

Vector Vector::operator/(int d){
	Vector v(x/d,-y/d,z/d);
	return v;
}

bool Vector::operator==(Vector other){
	return ((other.x==x) && (other.y==y) && (other.z==z));
}


Vector Vector::Normalize(Vector v)
{
	float length=sqrt((v.x*v.x)+(v.y*v.y)+(v.z*v.z));
	return Vector(v.x/length,v.y/length,v.z/length);
}