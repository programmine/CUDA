#ifndef WAVEMAP_H
#define WAVEMAP_H

class WMatrix;

/**
 * the height field of the water plane
 *
 * for the water surface simulation two buffers are needed
 * they are updated depending on the previous wave state and damping
 * after the update the two buffers are swapped
 * 
 */
class WaveMap
{
public:


	/// creates wave map px * py and a damping factor which is needed for the wave expansion
	WaveMap(int pX, int pY, float damp);
	WaveMap();

	/// swap two buffers
	void swap();

	/// changes height value at position (x,y).
	void push(int x, int y, float value);

	/// calculates current height field
	virtual void updateWaveMap();

	/// number of points along the x axis
	int getPointsX();

	/// number of points along the y axis
	int getPointsY();

	/// height at position (x,y)
	virtual float getHeight(int x, int y);

	virtual ~WaveMap(void);

protected:
	WMatrix* newWave;
	WMatrix* tmp;
	WMatrix* oldWave;
	
protected:
	int pointsX;
	int pointsY;
	
	float sizeX;
	float sizeY;

	float stepSize;
	
	float dampFactor;

};
#endif



