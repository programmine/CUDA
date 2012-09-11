
#include <windows.h>
#include <math.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <cstdlib>
#include <iostream> 
#include "wavemapCUDA.h"
#include "vector.h"
#include "waterplaneCUDA.h"

static unsigned long inKB(unsigned long bytes)
{ return bytes/1024; }

static unsigned long inMB(unsigned long bytes)
{ return bytes/(1024*1024); }

// Singleton
WaterPlane* WaterPlaneCUDA::getWaterPlane(){

	if(WaterPlaneExemplar == 0){
		WaterPlaneExemplar = new WaterPlaneCUDA;
	}
	return WaterPlaneExemplar;
}

WaterPlaneCUDA::WaterPlaneCUDA():WaterPlane(){

}

void WaterPlaneCUDA::configure(Vector upperLeft, Vector lowerRight, float dampFactor, float resolution)
{

	cudaSetDevice(cutGetMaxGflopsDeviceId());
	cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());


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


	this->stepSize = 1.0f/resolution;

	this->resolutionFactor = resolution;

	//reale Z - Achse ist x - Achse der WaterPlaneCUDA
	this->sizeX = (unsigned int) abs(upperLeft.z - lowerRight.z);

	//reale X -Achse ist y- Achse der WaterPlaneCUDA
	this->sizeY = (unsigned int) abs(upperLeft.x - lowerRight.x);

	//Anzahl der Netzpunkte in X -Richtung
	this->pointsX = (unsigned int)(sizeX * resolution);

	//Anzahl der Netzpunkte in Y -Richtung
	pointsY = (unsigned int)(sizeY * resolution);

	uLeft = upperLeft;

	lRight = lowerRight;

	//Der "Meeresspiegel"
	baseHeight = lRight.y;

	//Das Höhenfeld der WaterPlaneCUDA
	waveMap = NULL;

	initBuffer();

	gpu_newVertices = new float3[pointsX*pointsY];
	gpu_oldVertices = new float3[pointsX*pointsY];
	gpu_normals = new float3[pointsX*pointsY];

	for (int i=0;i<pointsX*pointsY;i++) {
		gpu_newVertices[i]=make_float3(0,0,0);
		gpu_oldVertices[i]=make_float3(0,0,0);
		gpu_normals[i]=make_float3(0,1.0,0);
	}
	cutilSafeCall(cudaMalloc((void**)&gpu_newVertices,pointsX*pointsY*sizeof(float3)));
	cutilSafeCall(cudaMalloc((void**)&gpu_oldVertices,pointsX*pointsY*sizeof(float3)));
	cutilSafeCall(cudaMalloc((void**)&gpu_normals,pointsX*pointsY*sizeof(float3)));

	drawMesh();
}

void WaterPlaneCUDA::disturbArea(float xmin, float zmin, float xmax, float zmax, float height)
{
	float radius = (float)(getWaterPlaneX(xmax-xmin))/2.0f;
	if (radius <= 0) radius = 1;
	float centerX = (float)getWaterPlaneX((xmax+xmin)/2.0f);
	float centerZ = (float)getWaterPlaneY((zmax+zmin)/2.0f);

	if ((((zmax+zmin)/2.0f) < this->uLeft.z) || (((zmax+zmin)/2.0f) > this->lRight.z)) return;
	if ((((xmax+xmin)/2.0f) < this->uLeft.x) || (((xmax+xmin)/2.0f) > this->lRight.x)) return;

	float r2 = radius * radius;

	int xminW = getWaterPlaneX(xmin);
	int zminW = getWaterPlaneY(zmin);
	int xmaxW = getWaterPlaneX(xmax);
	int zmaxW = getWaterPlaneY(zmax);

	if (xminW >= pointsX) xminW = pointsX-1;
	if (zminW >= pointsX) zminW = pointsX-1;
	if (xmaxW >= pointsX) xmaxW = pointsX-1;
	if (zmaxW >= pointsX) zmaxW = pointsX-1;

	if (xminW < 0) xminW = 0;
	if (zminW < 0) zminW = 0;
	if (xmaxW < 0) xmaxW = 0;
	if (zmaxW < 0) zmaxW = 0;

	glBindBuffer(GL_ARRAY_BUFFER, oldVertexBuffer);
	float3* verticesTest = (float3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);


	for(int x = xminW; x <= xmaxW; x++)
	{
		for (int y = zminW; y <= zmaxW; y++)
		{
			float insideCircle = ((x-centerX)*(x-centerX))+((y-centerZ)*(y-centerZ))-r2;

			if (insideCircle <= 0) {
				int vIndex = (y * pointsX) + x;
				if (vIndex < (pointsX*pointsY)) {
					verticesTest[vIndex].y = (insideCircle/r2)*height;
				}
			}
		} 
	}
	glUnmapBufferARB(GL_ARRAY_BUFFER);


}

void WaterPlaneCUDA::update()
{
	size_t num_bytes; 

	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_newVertex_resource, 0));
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gpu_newVertices, &num_bytes, cuda_newVertex_resource));
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_oldVertex_resource, 0));
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gpu_oldVertices, &num_bytes, cuda_oldVertex_resource));
	cutilSafeCall(cudaMemcpyToSymbol("DIM",&pointsX,sizeof(int)));
	updateWaveMapGPU1(gpu_newVertices,gpu_oldVertices);
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_newVertex_resource, 0));
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_oldVertex_resource, 0));
	
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_oldVertex_resource, 0));
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gpu_oldVertices, &num_bytes, cuda_oldVertex_resource));
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_normalsVB_resource, 0));
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gpu_normals, &num_bytes, cuda_normalsVB_resource));
	cutilSafeCall(cudaMemcpyToSymbol("DIM",&pointsX,sizeof(int)));
	updateNormalsGPU1(gpu_oldVertices,gpu_normals);
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_normalsVB_resource, 0));
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_oldVertex_resource, 0));

	//swap between old and new wave map
	struct cudaGraphicsResource *temp = cuda_oldVertex_resource;
	cuda_oldVertex_resource = cuda_newVertex_resource;
	cuda_newVertex_resource = temp;
}


void WaterPlaneCUDA::drawMesh()
{


	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	GLfloat mat_color1[] = { 0.2f, 0.6f, 0.9f };
	GLfloat mat_shininess =  110.0f ;
	GLfloat specRefl[] = {1.0f, 1.0f, 1.0f, 1.0f}; // default

	glEnableClientState( GL_VERTEX_ARRAY );
	glEnableClientState(GL_NORMAL_ARRAY);

	glBindBuffer( GL_ARRAY_BUFFER, newVertexBuffer );
	glVertexPointer( 3, GL_FLOAT, 0, (char *) NULL );

	glBindBuffer( GL_ARRAY_BUFFER, normalBuffer );
	glNormalPointer(GL_FLOAT, 0, (char *) NULL );
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVertexBuffer);

	if (showEdges)
	{
		glDisable(GL_LIGHTING);
		glColor3f(1,1,1);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawElements(GL_TRIANGLE_STRIP, ((pointsX*2)+2)*(pointsX-1), GL_UNSIGNED_INT, 0); 
	}
	glEnable(GL_LIGHTING);
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mat_color1);
	glMaterialfv(GL_FRONT, GL_SPECULAR, specRefl); 
	glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
	glDrawElements(GL_TRIANGLE_STRIP, ((pointsX*2)+2)*(pointsX-1), GL_UNSIGNED_INT, 0); 

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// Disable Pointers
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState( GL_VERTEX_ARRAY );// Disable Vertex Arrays
}



void WaterPlaneCUDA::initBuffer()
{
	//Start und Endkoordinaten für x-Richtung
	float startX = this->uLeft.x;
	float endX = this->lRight.x;

	//Start und Endkoordinaten für x-Richtung
	float startY = this->uLeft.z;
	float endY = this->lRight.z;

	cpu_newVertices = new float3[pointsX*pointsY];
	cpu_oldVertices = new float3[pointsX*pointsY];
	cpu_normals = new float3[pointsX*pointsY];
	unsigned int count = 0;
	for (float px = startX ; px< endX ; px+=stepSize){
		for (float py = startY; py < endY; py+=stepSize){
			cpu_newVertices[count].x = px;
			cpu_newVertices[count].y = 0;
			cpu_newVertices[count].z = py;
			cpu_oldVertices[count].x = px;
			cpu_oldVertices[count].y = 0;
			cpu_oldVertices[count].z = py;
			cpu_normals[count].x = 0;
			cpu_normals[count].y = 1;
			cpu_normals[count].z = 0;
			count++;
		}
	}


	createVBO(&newVertexBuffer, pointsX*pointsY*sizeof(float3));
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_newVertex_resource, newVertexBuffer, cudaGraphicsMapFlagsNone));
	createVBO(&oldVertexBuffer, pointsX*pointsY*sizeof(float3));
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_oldVertex_resource, oldVertexBuffer, cudaGraphicsMapFlagsNone));
	createVBO(&normalBuffer, pointsX*pointsY*sizeof(float3));
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_normalsVB_resource, normalBuffer, cudaGraphicsMapFlagsNone));
	createMeshIndexBuffer(&indexVertexBuffer, pointsX, pointsY);
	createMeshIndexBuffer(&indexNormalBuffer, pointsX, pointsY);

	glBindBuffer(GL_ARRAY_BUFFER, *&newVertexBuffer);
	glBufferData( GL_ARRAY_BUFFER, pointsY*pointsX*sizeof(float3), cpu_newVertices, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, *&oldVertexBuffer);
	glBufferData( GL_ARRAY_BUFFER, pointsY*pointsX*sizeof(float3), cpu_oldVertices, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, *&normalBuffer);
	glBufferData( GL_ARRAY_BUFFER, pointsY*pointsX*sizeof(float3), cpu_normals, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


WaterPlaneCUDA::~WaterPlaneCUDA(){
	cudaFree(gpu_normals);
	cudaFree(gpu_newVertices);
	cudaFree(gpu_oldVertices);
	cutilSafeCall(cudaGraphicsUnregisterResource(cuda_newVertex_resource));
	cutilSafeCall(cudaGraphicsUnregisterResource(cuda_oldVertex_resource));
	cutilSafeCall(cudaGraphicsUnregisterResource(cuda_normalsVB_resource));
	//cudaFree(DIM);
}