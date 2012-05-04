#include <cutil_inline.h>
#include <cuda.h> 

__global__ void updateWaveMapGPU2( float* dev_newWave, float* dev_oldWave,unsigned int *dev_arraySize ){
		int tid= blockIdx.x;
		while (tid < *dev_arraySize){
			unsigned int y = int(tid / 80);
			unsigned int x = tid % 80;
			unsigned int up = (x - 1) + (y * 80);
			unsigned int down = (x + 1) + (y * 80);
			unsigned int leftp = x + ((y - 1)* 80);
			unsigned int rightp = x + ((y + 1)* 80);


			float n = 0.0f;
			int no=1;
			if (x-1 >= 0) {
				n += dev_oldWave[up]; 
				no++;
			}
			if (x + 1 < 80) {
				n += dev_oldWave[down]; 
				no++;
			}
			if (y-1 >= 0) {
				n += dev_oldWave[leftp]; 
				no++;
			}
			if (y+1 < 80) {
				no++;
				n += dev_oldWave[rightp]; 
			}
			
			n = n/(float)no;
			n = n*2 - dev_newWave[tid];
			n = n - (n/32.0f);
			dev_newWave[tid] = n;
			tid += gridDim.x;
		}
}

void updateWaveMapGPU1(float* dev_newWave, float* dev_oldWave,unsigned int *dev_arraySize){
	updateWaveMapGPU2<<< 1 ,1 >>>(dev_newWave,dev_oldWave,dev_arraySize);
}

