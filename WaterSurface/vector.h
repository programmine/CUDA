#pragma once

class Vector
{
private:
	float vector[3];
	float normal[3];
	float texture[3];

public:
	Vector(float x, float y, float z);
	Vector(void);
	~Vector(void);
	float get(int pos);
	void setVector(float x, float y, float z);
	void set(int pos, float val);
	void setNormal(float x, float y, float z);
	float getNormal(int pos);
	void setTexture(float x, float y, float z);
	float getTexture(int pos);
	Vector crossProduct(Vector other);
	Vector & operator+(Vector&);       
	Vector & operator-(Vector&);    
	Vector & operator/(int); 
	bool operator==(Vector&);   
	static Vector Normalize(Vector);
	
};

