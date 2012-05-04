#ifndef WATERPLANE_H
#define WATERPLANE_H


class WaveMap;
class WMatrix;
class TriangleList;
#include "vector.h"
#include <vector>

/**
 * Water surface
 *
 * This class uses the functionality of the WaveMap to generate a surface mesh
 * The class is a singleton.
 *
 */
class WaterPlane
{
public:
	/**
	 * creates the next wave state
	 */
	virtual void update(void);

	/**
	 * returns instance of waterplane
	 */
	static WaterPlane* getWaterPlane();

	/**
	 * draws the mesh of the water plane
	 */
	virtual void drawMesh(void);

	/**
	 *
	 * Initialises the water plane
	 * 
	 * @param dampFactor expansion of the created waves
	 * @param resolution describes how many grid points are created per scaling unit
	 * 
	 */
	virtual void configure(Vector upperLeft, Vector lowerRight, float dampFactor, float resolution);

	/**
	 * method to disturb one point of surface.
	 *
	 * @param disturbingPoint one point to disturb surface.
	 * 
	 */
	void disturb(Vector disturbingPoint);

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
	void disturbArea(float xmin, float zmin, float xmax, float zmax, float height);

	/**
	*
	* show/hide triangle edges of mesh
	*
	**/
	void toggleEdges();

	~WaterPlane(void);

protected:

	/**
	 * Standard constructor
	 */
	WaterPlane();

	/**
	 * The WaterPlane Instance
	 */
	static WaterPlane* WaterPlaneExemplar;

	/** 
	 * Swaps the roles of the two buffers between old and new wave state
	 */
	void swapBuffers(void);

	/**
	 * Sets the height value of the given x,y (x,z) to the given depth
	 */
	void push(float x, float y, float depth);

	/**
	 * initialises the basic data structure
	 */
	void initBuffer(void);

	/**
	 * height on the position (x,y).
	 */
	int getHeight(int x, int y);

	/**
	 * calculates the water surface x point from the world coordinates of realX
	 */
	unsigned int getWaterPlaneX(float realX);

	/**
	 * calculates the water surface point y from the world coordinates of realY
	 */
	unsigned int getWaterPlaneY(float realY);

	/**
	 * creates triangle data structure of surface mesh
	 */
	void setupTriangleDataStructure(void);

	//resolution factor
	float resolutionFactor;

	//Vertices of the surface mesh
	std::vector<Vector*> vertices;

	/** 
	 * data structure (buffers) for wave creation
	 */
	WaveMap* waveMap;

	Vector uLeft;

	Vector lRight;

	unsigned int pointsX;

	unsigned int pointsY;

	unsigned int sizeX;

	unsigned int sizeY;

	float stepSize;

	float baseHeight;

	TriangleList *triangles;

	bool showEdges;
};

#endif
