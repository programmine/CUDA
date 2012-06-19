#include <cutil_inline.h>
#include <cuda.h> 

__global__ void updateWaveMapGPU2( float* dev_newWave, float* dev_oldWave,unsigned int *dev_arraySize, unsigned int *dev_arrayDIM){
		int tid= blockIdx.x;
		while (tid < *dev_arraySize){
			int y = int(tid / *dev_arrayDIM);
			int x = tid % *dev_arrayDIM;
			int up = (x - 1) + (y * *dev_arrayDIM);
			int down = (x + 1) + (y * *dev_arrayDIM);
			int leftp = x + ((y - 1)* *dev_arrayDIM);
			int rightp = x + ((y + 1)* *dev_arrayDIM);


			float n = 0.0f;
			int no=1;
			if (x-1 >= 0) {
				n += dev_oldWave[up]; 
				no++;
			}
			if (x + 1 < *dev_arrayDIM) {
				n += dev_oldWave[down]; 
				no++;
			}
			if (y-1 >= 0) {
				n += dev_oldWave[leftp]; 
				no++;
			}
			if (y+1 < *dev_arrayDIM) {
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

void updateWaveMapGPU1(float* dev_newWave, float* dev_oldWave,unsigned int *dev_arraySize, unsigned int *dev_arrayDIM){
	updateWaveMapGPU2<<< 5000 ,1 >>>(dev_newWave,dev_oldWave,dev_arraySize,dev_arrayDIM);
}

void updateNormalsGPU1(float3* dev_vertices, float3* dev_newNormals,unsigned int *dev_DIM){

}

__global__ void updateNormalsGPU2( float3* dev_vertices, float3* dev_newNormals,unsigned int *dev_DIM){
	
}

