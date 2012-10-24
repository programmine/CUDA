#ifndef DISTURBANCES_H
#define DISTURBANCES_H
#include "vector.h"

/// Save disturbed data to use in update
class Disturbances
{
public:
	Disturbances(void);
	Disturbances(float centerX,float centerZ, float radiusSQ, int xminW, int zminW, int xmaxW, int zmaxW, float height);
	~Disturbances(void);
	float radiusSQ;
	float centerX;
	float centerZ;
	int xminW;
	int zminW;
	int xmaxW;
	int zmaxW;
	float height;


};
#endif