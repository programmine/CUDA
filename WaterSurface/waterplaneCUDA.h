#ifndef WATERPLANECUDA_H
#define WATERPLANECUDA_H


class WaveMap;
class WMatrix;
#include "vector.h"
#include "waterplane.h"
#include <vector>
#include <cutil_inline.h>

/**
 * Water surface
 *
 * This class uses the functionality of the WaveMap to generate a surface mesh
 * The class is a singleton.
 *
 */
class WaterPlaneCUDA : public WaterPlane
{
public:
	/**
	 * creates the next wave state
	 */
	virtual void update(void);

	virtual void drawMesh(void);

	void drawTriangle(float3,float3,float3,float3,float3,float3);

	/**
	 * returns instance of WaterPlaneCUDA
	 */
	static WaterPlane* getWaterPlane();

	virtual void configure(Vector upperLeft, Vector lowerRight, float dampFactor, float resolution);

	virtual ~WaterPlaneCUDA(void);

protected:
	WaterPlaneCUDA();
	/**
	 * initialises the basic data structure
	 */
	void initBuffer(void);

	Vector calculateNormalFor4Neighbours(int,int);

	Vector calculateNormalFor8Neighbours(int,int);

	float3 *cpu_vertices;
	float3 *gpu_vertices;
	float3 *cpu_normals;
	float3 *gpu_normals;
	unsigned int *DIM;

};

#endif
