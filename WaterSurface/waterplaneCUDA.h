#ifndef WATERPLANECUDA_H
#define WATERPLANECUDA_H


class WMatrix;
class Disturbances;
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

	/**
	 * draws mesh
	 */
	virtual void drawMesh(void);

	/**
	 * returns instance of WaterPlaneCUDA
	 */
	static WaterPlane* getWaterPlane();

	/**
	 * method to disturb circular area of surface.
	 *
	 * @param xmin lower x value of aquare which contain circular area to disturb surface
	 * @param zmin lower z value of aquare which contain circular area to disturb surface
	 * @param xmax higher x value of aquare which contain circular area to disturb surface
	 * @param zmax higher z value of aquare which contain circular area to disturb surface
	 * @param height defines how heigh/low the circular area is pushed
	 * 
	 */
	virtual void disturbArea(float xmin, float zmin, float xmax, float zmax, float height);

	/**
	 *
	 * Initialises the water plane
	 * 
	 * @param upperLeft, lowerRight defines size of waterplane
	 * @param dampFactor expansion of the created waves
	 * @param resolution describes how many grid points are created per scaling unit
	 * 
	 */
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

	int timePassed;
	int timeSinceLast;

	float3 *cpu_newVertices;
	float3 *gpu_newVertices;
	float3 *cpu_oldVertices;
	float3 *gpu_oldVertices;
	float3 *cpu_normals;
	float3 *gpu_normals;
	std::vector<Disturbances*> disturbances;

};

#endif
