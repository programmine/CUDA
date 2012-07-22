
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


__device__ float3 calculateNormalFor4Neighbours(float3* vertices, unsigned int *DIM, int vIndex){
	int d = (int)(vIndex/ *DIM);

	/*     v4
		*--*--*
		| /|\ |
	v3  *--*--* v2
		| \|/ |
		*--*--*
		   v1
	*/
	float3 v = vertices[vIndex];
	float3 normal = make_float3(0,0,0);
	int v1Index = (vIndex - 1);
	int v2Index = (vIndex -  *DIM);
	int v3Index = (vIndex +  *DIM);
	int v4Index = (vIndex + 1);

	float3 v2,v1; 
	float3 s1,s2;

	if ((v1Index >= 0)&&((v1Index/ *DIM) == d)){

		//triangle - bottom and right
		if (v2Index >= 0){
			v2 = vertices[v1Index];
			v1 = vertices[v2Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s2,s1); 
		}

		//triangle - bottom and left
		if ((v3Index/ *DIM) <  *DIM){
			v2 = vertices[v1Index];
			v1 = vertices[v3Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s1,s2); 
		}
	}

	
	if ((v4Index/ *DIM) == d){

		//triangle - top and right
		if (v2Index >= 0){
			v2 = vertices[v2Index];
			v1 = vertices[v4Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s2,s1); 
		}

		//triangle - top and left
		if ((v3Index/ *DIM) <  *DIM){
			v2 = vertices[v3Index];
			v1 = vertices[v4Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s1,s2); 
		}
	}
	return normal;
}

__device__ float3 calculateNormalFor8Neighbours(float3* vertices, unsigned int *DIM, int vIndex){
	int d = (int)(vIndex/ *DIM);

	/*     v5
	v6	*--*--* v4
		|\ | /|
	v7  *--*--* v3
		|/ | \|
	v8	*--*--* v2
		   v1  
	*/

	float3 v = vertices[vIndex];
	float3 normal = make_float3(0,0,0);
	int v1Index = (vIndex - 1);
	int v2Index = (vIndex - *DIM - 1);
	int v3Index = (vIndex - *DIM);
	int v4Index = (vIndex - *DIM + 1);
	int v5Index = (vIndex + 1);
	int v6Index = (vIndex + *DIM + 1);
	int v7Index = (vIndex + *DIM);
	int v8Index = (vIndex + *DIM - 1);

	float3 v2,v1; 
	float3 s1,s2;
	if ((v1Index >= 0)&&((v1Index/ *DIM) == d)){

		//triangle - bottom and right
		if (v2Index >= 0){
			v2 = vertices[v1Index];
			v1 = vertices[v2Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s2,s1); 
		}

		//triangle - bottom and left
		if ((v8Index/ *DIM) < *DIM){
			v2 = vertices[v1Index];
			v1 = vertices[v8Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s1,s2); 
		}
	}

	
	if ((v3Index >= 0)&&(d > 0)){

		//triangle - middle right and bottom
		if ((v2Index >= 0)&&((v2Index/ *DIM) == d - 1)){
			v2 = vertices[v2Index];
			v1 = vertices[v3Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s2,s1); 
		}

		//triangle - middle right and top
		if ((v4Index >= 0)&&((v4Index/ *DIM) == d - 1)){
			v2 = vertices[v3Index];
			v1 = vertices[v4Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s1,s2); 
		}
	}

	if ((v5Index/ *DIM) == d){

		//triangle - top and right
		if ((v4Index >= 0)&&(d > 0)&&((v4Index/ *DIM) == d - 1)){
			v2 = vertices[v4Index];
			v1 = vertices[v5Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s2,s1); 
		}

		//triangle - top and left
		if ((d+1 <  *DIM)&&((v6Index/ *DIM) == d + 1)){
			v2 = vertices[v5Index];
			v1 = vertices[v6Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s2,s1); 
		}
	}

	if (v7Index <  *DIM){

		//triangle - left middle and top
		if ((d+1 <  *DIM)&&((v6Index/ *DIM) == d + 1)){
			v2 = vertices[v6Index];
			v1 = vertices[v7Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s2,s1); 
		}

		//triangle - left middle and bottom
		if ((d+1 <  *DIM)&&((v8Index/ *DIM) == d + 1)){
			v2 = vertices[v7Index];
			v1 = vertices[v8Index];
			s1 = v1 - v;
			s2 = v2 - v;
			normal = normal + cross(s1,s2); 
		}
	}

	return normal;
}


__global__ void updateNormalsGPU2( float3* dev_vertices, float3* dev_normals,unsigned int *dev_DIM){
	bool dimensionUneven = (*dev_DIM % 2 == 1);
	int vIndex;

	int tid= blockIdx.x;
	while (tid < (*dev_DIM* *dev_DIM)){

		vIndex = tid;

		int noOfNeighbourVertices = 0;
		
		if ((dimensionUneven)||((int)(vIndex / *dev_DIM))%2 == 0)
		{
			if (vIndex % 2 == 1)  noOfNeighbourVertices = 8;
			else noOfNeighbourVertices = 4;
		}
		else
		{
			if (vIndex % 2 == 1)  noOfNeighbourVertices = 4;
			else noOfNeighbourVertices = 8;
		}

		float3 normal = make_float3(0,1,0);
		
		if (noOfNeighbourVertices == 4) normal = calculateNormalFor4Neighbours(dev_vertices,dev_DIM,vIndex);
		else  normal = calculateNormalFor8Neighbours(dev_vertices,dev_DIM,vIndex);
		dev_normals[vIndex] = normalize(normal);
		tid += gridDim.x;
	}
}

void updateNormalsGPU1(float3* dev_vertices, float3* dev_normals,unsigned int *dev_DIM){
	updateNormalsGPU2<<<15000,1>>>(dev_vertices,dev_normals,dev_DIM);
}


