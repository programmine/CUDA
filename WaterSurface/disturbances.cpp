#include "disturbances.h"
#include <iostream>

Disturbances::Disturbances(void)
{
	
}

Disturbances::Disturbances(float centerX,float centerZ, float radiusSQ, int xminW, int zminW, int xmaxW, int zmaxW, float height)
{
	this->radiusSQ = radiusSQ;
	this->centerX = centerX;
	this->centerZ = centerZ;
	this->xminW = xminW;
	this->zminW = zminW;
	this->xmaxW = xmaxW;
	this->zmaxW = zmaxW;
	this->height = height;
}

Disturbances::~Disturbances(void)
{
}
