#include "waterplaneCUDA.h"
#include "wavemapCUDA.h"
#include "vector.h"
#include "trianglelistCUDA.h"
#include "triangle.h"
#include <windows.h>
#include <GL/gl.h>
#include <cstdlib>
#include <iostream> 
#include <math.h>

#include "GPUFunctions.cuh"

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

	stepSize = 1.0/resolution;

	resolutionFactor = resolution;

	//reale Z - Achse ist x - Achse der WaterPlaneCUDA
	sizeX = (unsigned int) abs(upperLeft.get(2) - lowerRight.get(2));

	//reale X -Achse ist y- Achse der WaterPlaneCUDA
	sizeY = (unsigned int) abs(upperLeft.get(0) - lowerRight.get(0));

	//Anzahl der Netzpunkte in X -Richtung
	pointsX = (unsigned int)(sizeX * resolution);

	//Anzahl der Netzpunkte in Y -Richtung
	pointsY = (unsigned int)(sizeY * resolution);

	uLeft = upperLeft;

	lRight = lowerRight;

	//Der "Meeresspiegel"
	baseHeight = lRight.get(1);

	triangles = new TriangleListCUDA();

	//Das Höhenfeld der WaterPlaneCUDA
	waveMap = new WaveMapCUDA(pointsX, pointsY, dampFactor);

	initBuffer();

	setupTriangleDataStructure();

	gpu_vertices = new float3[vertices.size()];
	gpu_normals = new float3[vertices.size()];
	cpu_normals = new float3[vertices.size()];

	for (unsigned int i=0;i<vertices.size();i++) {
		gpu_vertices[i]=make_float3(0,0,0);
		gpu_normals[i]=make_float3(0,1.0,0);
		cpu_normals[i]=make_float3(0,1.0,0);
	}
	cutilSafeCall(cudaMalloc((void**)&gpu_vertices,vertices.size()*sizeof(float3)));
	cutilSafeCall(cudaMalloc((void**)&gpu_normals,vertices.size()*sizeof(float3)));
	cutilSafeCall(cudaMalloc((void**)&DIM,sizeof(unsigned int)));

	drawMesh();
}

void WaterPlaneCUDA::setupTriangleDataStructure()
{
	bool odd = false;
	if (pointsY % 2 ==1) odd=true;

	for (unsigned int i=0; i < (pointsY*(pointsX-1));i++)
	{
		//every second column first row
		if (((i+1) % pointsY)==0) continue;
		if (((i / pointsY) % 2 == 0) || (odd))
		{
			if (i % 2 == 0)
			{	
				triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY)),(2*pointsX+1));
				triangles->AddTriangle(new Triangle(vertices.at(i+1),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));

			}else
			{
				triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY+1)),(2*pointsX+1));
				triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));
			}
		}
		else
		{
			if (i % 2 == 1)
			{	
				triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY)),(2*pointsX+1));
				triangles->AddTriangle(new Triangle(vertices.at(i+1),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));

			}else
			{

				triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY+1)),(2*pointsX+1));
				triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));
			}
		}

	}
}


void WaterPlaneCUDA::update()
{
	//update WaveMap
	waveMap->updateWaveMap();

	float n = 0.0;

	int vIndex = 0;

	int nIndex = 0;

	

	for (unsigned int x = 0; x< pointsX ; x++){

		for (unsigned int y = 0; y < pointsY; y++){

			//neuer Höhenwert
			n = waveMap->getHeight(x,y);

			n += this->baseHeight; 

			vIndex = (y * pointsX) + x;

			Vector *v = vertices.at(vIndex);
			v->set(1, n);
		}
	}


	cpu_vertices = convertFromCPUToGPUVertices();
	cutilSafeCall(cudaMemcpy(gpu_vertices,cpu_vertices,(pointsX*pointsX)*sizeof(float3),cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(gpu_normals,cpu_normals,(pointsX*pointsX)*sizeof(float3),cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(DIM,&pointsX,sizeof(unsigned int),cudaMemcpyHostToDevice));
	updateNormalsGPU1(gpu_vertices,gpu_normals,DIM);
	cutilSafeCall(cudaMemcpy(cpu_vertices,gpu_vertices,(pointsX*pointsX)*sizeof(float3),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(cpu_normals,gpu_normals,(pointsX*pointsX)*sizeof(float3),cudaMemcpyDeviceToHost));
	convertFromGPUToCPUVertices(cpu_vertices,cpu_normals);
}


float3* WaterPlaneCUDA::convertFromCPUToGPUVertices(){
	float3 *cvertices = new float3[vertices.size()];
	for (unsigned int i=0;i<vertices.size();i++) {
		cvertices[i]=make_float3(vertices.at(i)->get(0),vertices.at(i)->get(1),vertices.at(i)->get(2));
	}
	return cvertices;
}


void WaterPlaneCUDA::convertFromGPUToCPUVertices(float3* cvertices, float3* cnormals){
	for (unsigned int i=0;i<vertices.size();i++) {
		vertices.at(i)->setVector(cvertices[i].x,cvertices[i].y,cvertices[i].z);
		vertices.at(i)->setNormal(cnormals[i].x,cnormals[i].y,cnormals[i].z);
	}
}


Vector WaterPlaneCUDA::calculateNormalFor8Neighbours(int vIndex, int Dimension)
{
	int d = (int)(vIndex/Dimension);

	/*     v5
	v6	*--*--* v4
		|\ | /|
	v7  *--*--* v3
		|/ | \|
	v8	*--*--* v2
		   v1  
	*/

	Vector *v = vertices.at(vIndex);
	Vector *normal = new Vector(0,0,0);
	int v1Index = (vIndex - 1);
	int v2Index = (vIndex - Dimension - 1);
	int v3Index = (vIndex - Dimension);
	int v4Index = (vIndex - Dimension + 1);
	int v5Index = (vIndex + 1);
	int v6Index = (vIndex + Dimension + 1);
	int v7Index = (vIndex + Dimension);
	int v8Index = (vIndex + Dimension - 1);

	Vector *v2,*v1; 
	Vector s1,s2;
	if ((v1Index >= 0)&&((v1Index/Dimension) == d)){

		//triangle - bottom and right
		if (v2Index >= 0){
			v2 = vertices.at(v1Index);
			v1 = vertices.at(v2Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s2.crossProduct(s1)); 
		}

		//triangle - bottom and left
		if ((v8Index/Dimension) < Dimension){
			v2 = vertices.at(v1Index);
			v1 = vertices.at(v8Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s1.crossProduct(s2)); 
		}
	}

	
	if ((v3Index >= 0)&&(d > 0)){

		//triangle - middle right and bottom
		if ((v2Index >= 0)&&((v2Index/Dimension) == d - 1)){
			v2 = vertices.at(v2Index);
			v1 = vertices.at(v3Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s2.crossProduct(s1)); 
		}

		//triangle - middle right and top
		if ((v4Index >= 0)&&((v4Index/Dimension) == d - 1)){
			v2 = vertices.at(v3Index);
			v1 = vertices.at(v4Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s1.crossProduct(s2)); 
		}
	}

	if ((v5Index/Dimension) == d){

		//triangle - top and right
		if ((v4Index >= 0)&&(d > 0)&&((v4Index/Dimension) == d - 1)){
			v2 = vertices.at(v4Index);
			v1 = vertices.at(v5Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s2.crossProduct(s1)); 
		}

		//triangle - top and left
		if ((d+1 < Dimension)&&((v6Index/Dimension) == d + 1)){
			v2 = vertices.at(v5Index);
			v1 = vertices.at(v6Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s2.crossProduct(s1)); 
		}
	}

	if (v7Index < Dimension){

		//triangle - left middle and top
		if ((d+1 < Dimension)&&((v6Index/Dimension) == d + 1)){
			v2 = vertices.at(v6Index);
			v1 = vertices.at(v7Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s2.crossProduct(s1)); 
		}

		//triangle - left middle and bottom
		if ((d+1 < Dimension)&&((v8Index/Dimension) == d + 1)){
			v2 = vertices.at(v7Index);
			v1 = vertices.at(v8Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s1.crossProduct(s2)); 
		}
	}

	return *normal;
}


Vector WaterPlaneCUDA::calculateNormalFor4Neighbours(int vIndex, int Dimension)
{
	int d = (int)(vIndex/Dimension);

	/*     v4
		*--*--*
		| /|\ |
	v3  *--*--* v2
		| \|/ |
		*--*--*
		   v1
	*/
	Vector *v = vertices.at(vIndex);
	Vector *normal = new Vector(0,0,0);
	int v1Index = (vIndex - 1);
	int v2Index = (vIndex - Dimension);
	int v3Index = (vIndex + Dimension);
	int v4Index = (vIndex + 1);

	Vector *v2,*v1; 
	Vector s1,s2;

	if ((v1Index >= 0)&&((v1Index/Dimension) == d)){

		//triangle - bottom and right
		if (v2Index >= 0){
			v2 = vertices.at(v1Index);
			v1 = vertices.at(v2Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s2.crossProduct(s1)); 
		}

		//triangle - bottom and left
		if ((v3Index/Dimension) < Dimension){
			v2 = vertices.at(v1Index);
			v1 = vertices.at(v3Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s1.crossProduct(s2)); 
		}
	}

	
	if ((v4Index/Dimension) == d){

		//triangle - top and right
		if (v2Index >= 0){
			v2 = vertices.at(v2Index);
			v1 = vertices.at(v4Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s2.crossProduct(s1)); 
		}

		//triangle - top and left
		if ((v3Index/Dimension) < Dimension){
			v2 = vertices.at(v3Index);
			v1 = vertices.at(v4Index);
			s1 = *v1 - *v;
			s2 = *v2 - *v;
			*normal = *normal + (s1.crossProduct(s2)); 
		}
	}
	return *normal;
}

void WaterPlaneCUDA::initBuffer()
{
	vertices.clear();

	//Start und Endkoordinaten für x-Richtung
	float startX = this->uLeft.get(0);
	float endX = this->lRight.get(0);

	//Start und Endkoordinaten für x-Richtung
	float startY = this->uLeft.get(2);
	float endY = this->lRight.get(2);

	//erzeuge entsprechend der Auflösung die benötigten Anzahl an Vertices und
	// die dazugehörigen Normalen
	for (float px = startX ; px< endX ; px+=stepSize){

		for (float py = startY; py < endY; py+=stepSize){

			//Vertex
			Vector *v = new Vector(px, this->baseHeight, py);
			v->setNormal(0,1.0,0);
			vertices.push_back(v);
		}
	}
}


WaterPlaneCUDA::~WaterPlaneCUDA(){
	cudaFree(gpu_normals);
	cudaFree(gpu_vertices);
	cudaFree(DIM);
}