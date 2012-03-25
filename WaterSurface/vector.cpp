#include "vector.h"
#include <iostream> 
#include <math.h>

Vector::Vector(float x, float y, float z)
{
	vector[0]=x;
	vector[1]=y;
	vector[2]=z;
}

Vector::Vector()
{
	vector[0]=0;
	vector[1]=0;
	vector[2]=0;
}

Vector::~Vector(void)
{
}

float Vector::get(int pos){
	return vector[pos];
}

void Vector::setVector(float x, float y, float z){
	vector[0]=x;
	vector[1]=y;
	vector[2]=z;
}

void Vector::set(int pos, float val){
	vector[pos]=val;
}

Vector Vector::crossProduct(Vector other)
{
	Vector cross(0.0f,0.0f,0.0f);
	cross.set(0, this->get(1) * other.get(2) - other.get(1) * this->get(2));
	cross.set(1, this->get(2) * other.get(0) - other.get(2) * this->get(0));
	cross.set(2, this->get(0) * other.get(1) - other.get(0) * this->get(1));
	return cross;
}

void Vector::setNormal(float x, float y, float z){
	normal[0]=x;
	normal[1]=y;
	normal[2]=z;
}

float Vector::getNormal(int pos){
	return normal[pos];
}

void Vector::setTexture(float x, float y, float z){
	texture[0]=x;
	texture[1]=y;
	texture[2]=z;
}

float Vector::getTexture(int pos){
	return texture[pos];
}

Vector& Vector::operator+(Vector& other){
	Vector v(other.get(0)+vector[0],other.get(1)+vector[1],other.get(2)+vector[2]);
	return v;
}


Vector& Vector::operator-(Vector& other){
	Vector v(-other.get(0)+vector[0],-other.get(1)+vector[1],-other.get(2)+vector[2]);
	return v;
}

Vector& Vector::operator/(int d){
	Vector v(vector[0]/d,-vector[1]/d,vector[2]/d);
	return v;
}

bool Vector::operator==(Vector& other){
	return ((other.get(0)==vector[0]) && (other.get(1)==vector[1]) && (other.get(2)==vector[2]));
}


Vector Vector::Normalize(Vector v)
{
	float length=sqrt((v.get(0)*v.get(0))+(v.get(1)*v.get(1))+(v.get(2)*v.get(2)));
	return Vector(v.get(0)/length,v.get(1)/length,v.get(2)/length);
}