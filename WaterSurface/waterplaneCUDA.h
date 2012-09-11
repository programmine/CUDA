#ifndef WATERPLANECUDA_H
#define WATERPLANECUDA_H


class WaveMap;
class WMatrix;
#include "vector.h"
#include "waterplane.h"
#include <vector>
#include <cutil_inline.h>
#include "GPUFunctions.cuh"

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

	/**
	 * returns instance of WaterPlaneCUDA
	 */
	static WaterPlane* getWaterPlane();

	virtual void disturbArea(float xmin, float zmin, float xmax, float zmax, float height);

	virtual void configure(Vector upperLeft, Vector lowerRight, float dampFactor, float resolution);

	~WaterPlaneCUDA(void);

protected:
	WaterPlaneCUDA();
	/**
	 * initialises the basic data structure
	 */
	void initBuffer(void);

	struct cudaGraphicsResource *cuda_newVertex_resource, *cuda_oldVertex_resource, *cuda_normalsVB_resource; // handles OpenGL-CUDA exchange
	GLuint indexVertexBuffer;
	GLuint newVertexBuffer;
	GLuint oldVertexBuffer;

	GLuint indexNormalBuffer;
	GLuint normalBuffer;

	float3 *cpu_newVertices;
	float3 *gpu_newVertices;
	float3 *cpu_oldVertices;
	float3 *gpu_oldVertices;
	float3 *cpu_normals;
	float3 *gpu_normals;

};

#endif
