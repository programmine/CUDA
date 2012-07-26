
#include <cuda.h>

#include <cutil_inline.h>
#include <cutil_math.h>

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


__global__ void updateNormalsGPU2( float3* dev_vertices, float3* dev_normals,unsigned int *dev_DIM){

	int nIndex;
	int tid= blockIdx.x;
	
	
	while (tid < (*dev_DIM* *dev_DIM)){

	/*
			   1  2
			*--*--*
			| /| /|
		  6 *--*--* 3
			| /| /|
			*--*--*
			5  4
*/

		nIndex = tid;
		int d = (int)(nIndex/ *dev_DIM);
		
		int v4Index = (nIndex - 1);
		int v3Index = (nIndex - *dev_DIM);
		int v2Index = (nIndex - *dev_DIM + 1);
		int v1Index = (nIndex + 1);
		int v6Index = (nIndex + *dev_DIM);
		int v5Index = (nIndex + *dev_DIM - 1);
		
		float3 vertex = dev_vertices[nIndex];
		float3 newNormal = make_float3(0,0,0);
		
		
		
		if ((v1Index/ *dev_DIM) == d){
			float3 v1 = dev_vertices[v1Index];
			float3 s1 = v1 - vertex;
			if ((v2Index >= 0)&&(d > 0)&&((v2Index/ *dev_DIM) == d - 1)){
				float3 v2 = dev_vertices[v2Index];
				float3 s2 = v2 - vertex;
				newNormal = newNormal + cross(s2,s1);
			}
			
			if (v6Index <  *dev_DIM){
				float3 v6 = dev_vertices[v6Index];
				float3 s6 = v6 - vertex;
				newNormal = newNormal + cross(s1,s6);
			}
		}
		if ((v4Index >= 0)&&((v4Index/ *dev_DIM) == d)){
			float3 v4 = dev_vertices[v4Index];
			float3 s4 = v4 - vertex;
			if ((v5Index/ *dev_DIM) < *dev_DIM){
				float3 v5 = dev_vertices[v5Index];
				float3 s5 = v5 - vertex;
				newNormal = newNormal + cross(s5,s4);
			}

			if ((v3Index >= 0)&&(d > 0)){
				float3 v3 = dev_vertices[v3Index];
				float3 s3 = v3 - vertex;
				newNormal = newNormal + cross(s4,s3);
			}
		}
		dev_normals[nIndex] = normalize(newNormal);
		tid += gridDim.x;
	}
	
	
	
	
	
}

void updateNormalsGPU1(float3* dev_vertices, float3* dev_normals,unsigned int *dev_DIM){
	updateNormalsGPU2<<<15000,1>>>(dev_vertices,dev_normals,dev_DIM);
}

