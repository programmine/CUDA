#ifndef WATERPLANECUDA_H
#define WATERPLANECUDA_H


class WaveMap;
class WMatrix;
class TriangleList;
#include "vector.h"
#include "waterplane.h"
#include <vector>

/**
 * Water surface
 *
 * This class uses the functionality of the WaveMap to generate a surface mesh
 * The class is a singleton.
 *
 */
class WaterPlaneCUDA : WaterPlane
{
public:
	/**
	 * creates the next wave state
	 */
	void update(void);

	/**
	 * returns instance of WaterPlaneCUDA
	 */
	static WaterPlane* getWaterPlane();

	/**
	 *
	 * Initialises the water plane
	 * 
	 * @param dampFactor expansion of the created waves
	 * @param resolution describes how many grid points are created per scaling unit
	 * 
	 */
	void configure(Vector upperLeft, Vector lowerRight, float dampFactor, float resolution);

protected:

	/**
	 * initialises the basic data structure
	 */
	void initBuffer(void);

	/**
	 * creates triangle data structure of surface mesh
	 */
	void setupTriangleDataStructure(void);

};

#endif
