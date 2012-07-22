#pragma once

class Vector
{
public:
	float x;
	float y;
	float z;
	Vector();
	Vector(float,float,float);
	~Vector();
	Vector crossProduct(Vector other);
	Vector  operator+(Vector);       
	Vector  operator-(Vector);    
	Vector  operator/(int); 
	bool operator==(Vector);   
	static Vector Normalize(Vector);
	
};

