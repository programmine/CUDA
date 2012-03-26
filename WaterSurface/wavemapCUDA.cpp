#include "wavemapCUDA.h"
#include "wmatrix.h"

#include <iostream>

void WaveMapCUDA::updateWaveMap()
{

	float n = 0.0;

	/* Skip the edges to allow area sampling*/
	for(int x = 0; x < pointsX; x++)
	{
		for (int y = 0; y < pointsY; y++)
		{
			//my forumla

			n = 0;
			int no=0;
			if (x-1 >= 0) {
				n += oldWave->getElement ( x - 1, y); 
				no++;
			}
			if (x+1 < pointsX) {
				n += oldWave->getElement ( x + 1, y);
				no++;
			}
			if (y-1 >= 0) {
				n += oldWave->getElement ( x , y - 1);
				no++;
			}
			if (y+1 < pointsY) {
				no++;
				n += oldWave->getElement ( x , y + 1);
			}
			
			n /= no;

			n = n*2 - newWave->getElement(x,y);

			n = n - (n/dampFactor);

			newWave->setElement(x, y, n);
		}
	}
	this->swap();
}

