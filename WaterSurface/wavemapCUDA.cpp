#include <cutil_inline.h>
#include <cuda.h>
#include "wavemapCUDA.h"
#include "wmatrixcuda.h"
#include <iostream>
#include <stdio.h>

#include "GPUFunctions.cuh"

#define N 10



static unsigned long inKB(unsigned long bytes)
{ return bytes/1024; }

static unsigned long inMB(unsigned long bytes)
{ return bytes/(1024*1024); }

WaveMapCUDA::WaveMapCUDA(unsigned int pX, unsigned int pY, float damp)
{
	this->oldWave = new WMatrix(pX, pY);

	this->newWave = new WMatrix(pX, pY);

	this->dampFactor = damp;

	this->pointsX = pX;

	this->pointsY = pY;

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



	*rowSize = pY;
	*arraySize = pY*pY;
	for (unsigned int i=0;i<*arraySize;i++) {
		dev_oldWave[i]=0;
		dev_newWave[i]=0;
	}
	cutilSafeCall(cudaMalloc((void**)&dev_oldWave,pY*pY*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&dev_newWave,pY*pY*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&dev_arraySize,sizeof(unsigned int)));
	cutilSafeCall(cudaMalloc((void**)&dev_arrayDIM,sizeof(unsigned int)));
	
}


void WaveMapCUDA::updateWaveMap()
{
	//cutilSafeCall(cudaMemcpy(dev_newWave,newWave->Values,(*arraySize)*sizeof(float),cudaMemcpyHostToDevice));
	//cutilSafeCall(cudaMemcpy(dev_oldWave,oldWave->Values,(*arraySize)*sizeof(float),cudaMemcpyHostToDevice));
	//cutilSafeCall(cudaMemcpy(dev_arraySize,this->arraySize,sizeof(unsigned int),cudaMemcpyHostToDevice));
	//cutilSafeCall(cudaMemcpy(dev_arrayDIM,this->rowSize,sizeof(unsigned int),cudaMemcpyHostToDevice));
	//updateWaveMapGPU1(dev_newWave,dev_oldWave, dev_arrayDIM);
	//cutilSafeCall(cudaMemcpy(newWave->Values,dev_newWave,(*arraySize)*sizeof(float),cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(oldWave->Values,dev_oldWave,(*arraySize)*sizeof(float),cudaMemcpyDeviceToHost));

	//this->swap();
}


WaveMapCUDA::~WaveMapCUDA(){
	cudaFree(dev_newWave);
	cudaFree(dev_oldWave);
	cudaFree(dev_arraySize);
	cudaFree(dev_arrayDIM);
}


