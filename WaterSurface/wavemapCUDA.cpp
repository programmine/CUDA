#include <cutil_inline.h>
#include <cuda.h>
#include "wavemapCUDA.h"
#include "wmatrix.h"
#include <iostream>
#include <stdio.h>

#include "GPUFunctions.cuh"

#define N 10



static unsigned long inKB(unsigned long bytes)
{ return bytes/1024; }

static unsigned long inMB(unsigned long bytes)
{ return bytes/(1024*1024); }



WaveMapCUDA::WaveMapCUDA(unsigned int pX, unsigned int pY, float damp):WaveMap(pX,pY,damp)
{
	rowSize =  new unsigned int();
	arraySize =  new unsigned int();
	dev_arraySize =  new unsigned int();
	dev_oldWave = new float[pY*pY];
	dev_newWave = new float[pY*pY];

	unsigned int free, total;
	int gpuCount, i;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;

	cuInit(0);

	cuDeviceGetCount(&gpuCount);
	printf("Detected %d GPU\n",gpuCount);

	for (i=0; i<gpuCount; i++)
	{
		cuDeviceGet(&dev,i);
		cuCtxCreate(&ctx, 0, dev);
		res = cuMemGetInfo(&free, &total);
		if(res != CUDA_SUCCESS)
			printf("!!!! cuMemGetInfo failed! (status = %x)", res);
		printf("^^^^ Device: %d\n",i);
		printf("^^^^ Free : %lu bytes (%lu KB) (%lu MB)\n", free, inKB(free), inMB(free));
		printf("^^^^ Total: %lu bytes (%lu KB) (%lu MB)\n", total, inKB(total), inMB(total));
		printf("^^^^ %f%% free, %f%% used\n", 100.0*free/(double)total, 100.0*(total - free)/(double)total);
		cuCtxDetach(ctx);
	}


	*rowSize = pY;
	*arraySize = pY*pY;
	for (unsigned int i=0;i<*arraySize;i++) {
		dev_oldWave[i]=0;
		dev_newWave[i]=0;
	}
	cutilSafeCall(cudaMalloc((void**)&dev_oldWave,pY*pY*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&dev_newWave,pY*pY*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&dev_arraySize,sizeof(unsigned int)));
	
}

float* WaveMapCUDA::From2DTo1D(float** array2D, unsigned int size, float* array1D) {
	for (unsigned int column = 0; column < size; column++){
		for (unsigned int row = 0; row < size; row++){
			unsigned int index = row+(column*size);
			array1D[index] = array2D[column][row];
		}
	}
	return array1D;
}

float** WaveMapCUDA::From1DTo2D(float* array1D, unsigned int size, float** array2D) {
	for (unsigned int i = 0; i< size; i++){
		array2D[i] = new float[size];
	}

	for (unsigned int index = 0; index < (size*size); index++){
			unsigned int column = int(index/size);
			unsigned int row = index % size;
			array2D[column][row] = array1D[index];
	}
	return array2D;
}

void WaveMapCUDA::updateWaveMap()
{
	float* dev_newWave2 = new float[*arraySize]();
	float* dev_oldWave2 = new float[*arraySize]();
	From2DTo1D(newWave->Values,*rowSize, dev_newWave2);
	From2DTo1D(oldWave->Values,*rowSize, dev_oldWave2);
	cutilSafeCall(cudaMemcpy(dev_newWave,dev_newWave2,(*arraySize)*sizeof(float),cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(dev_oldWave,dev_oldWave2,(*arraySize)*sizeof(float),cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(dev_arraySize,this->arraySize,sizeof(unsigned int),cudaMemcpyHostToDevice));
	updateWaveMapGPU1(dev_newWave,dev_oldWave, dev_arraySize);
	cutilSafeCall(cudaMemcpy(dev_newWave2,dev_newWave,(*arraySize)*sizeof(float),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(dev_oldWave2,dev_oldWave,(*arraySize)*sizeof(float),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(arraySize,dev_arraySize,sizeof(unsigned int),cudaMemcpyDeviceToHost));
	From1DTo2D(dev_newWave2,*rowSize, newWave->Values);
	From1DTo2D(dev_oldWave2,*rowSize, oldWave->Values);
	this->swap();
}


WaveMapCUDA::~WaveMapCUDA(){
	cudaFree(dev_newWave);
	cudaFree(dev_oldWave);
	cudaFree(dev_arraySize);
}


