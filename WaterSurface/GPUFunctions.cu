
#include <cuda.h>

#include <cutil_inline.h>
#include <cutil_math.h>

__device__ int DIM;

__global__ void updateWaveMapGPU2( float3* dev_newWave, float3* dev_oldWave){
		int tid = threadIdx.x + (blockIdx.x * blockDim.x);
		while (tid < (DIM* DIM)){
		
			int y = int(tid / DIM);
			int x = tid % DIM;
			int up = (x - 1) + (y * DIM);
			int down = (x + 1) + (y * DIM);
			int leftp = x + ((y - 1)* DIM);
			int rightp = x + ((y + 1)* DIM);


			float n = 0.0f;
			int no=0;
			if (x-1 >= 0) {
				n += dev_oldWave[up].y; 
				no++;
			}
			if (x + 1 < DIM) {
				n += dev_oldWave[down].y; 
				no++;
			}
			if (y-1 >= 0) {
				n += dev_oldWave[leftp].y; 
				no++;
			}
			if (y+1 < DIM) {
				no++;
				n += dev_oldWave[rightp].y; 
			}
			
			n = n/(float)no;
			n = (n*2) - dev_newWave[tid].y;
			n = n - ((n/32.0f));
			dev_newWave[tid].y = n;
			tid += gridDim.x*blockDim.x;
		}
}

void updateWaveMapGPU1(float3* dev_newWave, float3* dev_oldWave){
	updateWaveMapGPU2<<< 50,50 >>>(dev_newWave,dev_oldWave);
}



__global__ void updateNormalsGPU2( float3* dev_vertices, float3* dev_normals){

	int nIndex;
	int tid= threadIdx.x + blockIdx.x * blockDim.x;
	
	
	while (tid < (DIM* DIM)){

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
		int d = (int)(nIndex/ DIM);
		
		int v4Index = (nIndex - 1);
		int v3Index = (nIndex - DIM);
		int v2Index = (nIndex - DIM + 1);
		int v1Index = (nIndex + 1);
		int v6Index = (nIndex + DIM);
		int v5Index = (nIndex + DIM - 1);
		
		float3 vertex = dev_vertices[nIndex];
		float3 newNormal = make_float3(0,0,0);
		
		
		
		if ((v1Index/ DIM) == d){
			float3 v1 = dev_vertices[v1Index];
			float3 s1 = v1 - vertex;
			if ((v2Index >= 0)&&(d > 0)&&((v2Index/ DIM) == d - 1)){
				float3 v2 = dev_vertices[v2Index];
				float3 s2 = v2 - vertex;
				newNormal = newNormal + cross(s2,s1);
			}
			
			if (v6Index <  DIM){
				float3 v6 = dev_vertices[v6Index];
				float3 s6 = v6 - vertex;
				newNormal = newNormal + cross(s1,s6);
			}
		}
		if ((v4Index >= 0)&&((v4Index/ DIM) == d)){
			float3 v4 = dev_vertices[v4Index];
			float3 s4 = v4 - vertex;
			if ((v5Index/ DIM) < DIM){
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
		tid += gridDim.x*blockDim.x;
	}
	
	
	
	
	
}

void updateNormalsGPU1(float3* dev_vertices, float3* dev_normals){
	updateNormalsGPU2<<<50,50>>>(dev_vertices,dev_normals);
}

