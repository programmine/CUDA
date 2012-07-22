#include "wavemapCUDA.h"
#include "vector.h"
#include "trianglelistCUDA.h"
#include "triangle.h"
#include "waterplaneCUDA.h"
#include "GPUFunctions.cuh"
#include <windows.h>
#include <math.h>
#include <GL/glew.h>
#include <cstdlib>
#include <iostream> 




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
	sizeX = (unsigned int) abs(upperLeft.z - lowerRight.z);

	//reale X -Achse ist y- Achse der WaterPlaneCUDA
	sizeY = (unsigned int) abs(upperLeft.x - lowerRight.x);

	//Anzahl der Netzpunkte in X -Richtung
	pointsX = (unsigned int)(sizeX * resolution);

	//Anzahl der Netzpunkte in Y -Richtung
	pointsY = (unsigned int)(sizeY * resolution);

	uLeft = upperLeft;

	lRight = lowerRight;

	//Der "Meeresspiegel"
	baseHeight = lRight.y;

	//Das Höhenfeld der WaterPlaneCUDA
	waveMap = new WaveMapCUDA(pointsX, pointsY, dampFactor);

	initBuffer();

	//setupTriangleDataStructure();

	gpu_vertices = new float3[pointsX*pointsY];
	gpu_normals = new float3[pointsX*pointsY];

	for (unsigned int i=0;i<pointsX*pointsY;i++) {
		gpu_vertices[i]=make_float3(0,0,0);
		gpu_normals[i]=make_float3(0,1.0,0);
	}
	cutilSafeCall(cudaMalloc((void**)&gpu_vertices,pointsX*pointsY*sizeof(float3)));
	cutilSafeCall(cudaMalloc((void**)&gpu_normals,pointsX*pointsY*sizeof(float3)));
	cutilSafeCall(cudaMalloc((void**)&DIM,sizeof(unsigned int)));

	drawMesh();
}

void WaterPlaneCUDA::setupTriangleDataStructure()
{
	//bool odd = false;
	//if (pointsY % 2 ==1) odd=true;

	//for (unsigned int i=0; i < (pointsY*(pointsX-1));i++)
	//{
	//	//every second column first row
	//	if (((i+1) % pointsY)==0) continue;
	//	if (((i / pointsY) % 2 == 0) || (odd))
	//	{
	//		if (i % 2 == 0)
	//		{	
	//			triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY)),(2*pointsX+1));
	//			triangles->AddTriangle(new Triangle(vertices.at(i+1),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));

	//		}else
	//		{
	//			triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY+1)),(2*pointsX+1));
	//			triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));
	//		}
	//	}
	//	else
	//	{
	//		if (i % 2 == 1)
	//		{	
	//			triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY)),(2*pointsX+1));
	//			triangles->AddTriangle(new Triangle(vertices.at(i+1),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));

	//		}else
	//		{

	//			triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY+1)),(2*pointsX+1));
	//			triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));
	//		}
	//	}

	//}
}


void WaterPlaneCUDA::update()
{
	//update WaveMap
	waveMap->updateWaveMap();

	float n = 0.0;

	int vIndex = 0;

	int nIndex = 0;

	

	//for (unsigned int x = 0; x< pointsX ; x++){

	//	for (unsigned int y = 0; y < pointsY; y++){

	//		//neuer Höhenwert
	//		n = waveMap->getHeight(x,y);

	//		n += this->baseHeight; 

	//		vIndex = (y * pointsX) + x;

	//		Vector *v = vertices.at(vIndex);
	//		v->set(1, n);
	//	}
	//}

	cutilSafeCall(cudaMemcpy(gpu_vertices,cpu_vertices,(pointsX*pointsX)*sizeof(float3),cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(gpu_normals,cpu_normals,(pointsX*pointsX)*sizeof(float3),cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(gpu_normals,CUDAnormals,(pointsX*pointsX)*sizeof(float3),cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(DIM,&pointsX,sizeof(unsigned int),cudaMemcpyHostToDevice));
	updateNormalsGPU1(gpu_vertices,gpu_normals,DIM);
	cutilSafeCall(cudaMemcpy(CUDAvertices,gpu_vertices,(pointsX*pointsX)*sizeof(float3),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(CUDAnormals,gpu_normals,(pointsX*pointsX)*sizeof(float3),cudaMemcpyDeviceToHost));
}


void WaterPlaneCUDA::drawMesh()
{
	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


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
				drawTriangle(CUDAvertices[i],CUDAvertices[i+1],CUDAvertices[i+pointsY],CUDAnormals[i],CUDAnormals[i+1],CUDAnormals[i+pointsY]);
				//triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY)),(2*pointsX+1));
				drawTriangle(CUDAvertices[i+1],CUDAvertices[i+pointsY+1],CUDAvertices[i+pointsY],CUDAnormals[i+1],CUDAnormals[i+pointsY+1],CUDAnormals[i+pointsY]);
				//triangles->AddTriangle(new Triangle(vertices.at(i+1),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));

			}else
			{
				drawTriangle(CUDAvertices[i],CUDAvertices[i+1],CUDAvertices[i+pointsY+1],CUDAnormals[i],CUDAnormals[i+1],CUDAnormals[i+pointsY+1]);
				//triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY+1)),(2*pointsX+1));
				drawTriangle(CUDAvertices[i],CUDAvertices[i+pointsY+1],CUDAvertices[i+pointsY],CUDAnormals[i],CUDAnormals[i+pointsY+1],CUDAnormals[i+pointsY]);
				//triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));
			}
		}
		else
		{
			if (i % 2 == 1)
			{	
				drawTriangle(CUDAvertices[i],CUDAvertices[i+1],CUDAvertices[i+pointsY],CUDAnormals[i],CUDAnormals[i+1],CUDAnormals[i+pointsY]);
				//triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY)),(2*pointsX+1));
				drawTriangle(CUDAvertices[i+1],CUDAvertices[i+pointsY+1],CUDAvertices[i+pointsY],CUDAnormals[i+1],CUDAnormals[i+pointsY+1],CUDAnormals[i+pointsY]);
				//triangles->AddTriangle(new Triangle(vertices.at(i+1),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));

			}else
			{
				drawTriangle(CUDAvertices[i],CUDAvertices[i+1],CUDAvertices[i+pointsY+1],CUDAnormals[i],CUDAnormals[i+1],CUDAnormals[i+pointsY+1]);
				//triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+1),vertices.at(i+pointsY+1)),(2*pointsX+1));
				drawTriangle(CUDAvertices[i],CUDAvertices[i+pointsY+1],CUDAvertices[i+pointsY],CUDAnormals[i],CUDAnormals[i+pointsY+1],CUDAnormals[i+pointsY]);
				//triangles->AddTriangle(new Triangle(vertices.at(i),vertices.at(i+pointsY+1),vertices.at(i+pointsY)),(2*pointsX+1));
			}
		}

	}


	//for (int triangleIndex = 0; triangleIndex < triangles->GetCount(); triangleIndex++)
	//{
	//	Triangle *tr = triangles->GetTriangle(triangleIndex);
	//	if (showEdges)
	//	{
	//		glDisable(GL_LIGHTING);
	//		glColor3f(1,1,1);
	//		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//		glBegin(GL_TRIANGLES);
	//		glVertex3f(tr->Point1->x,tr->Point1->y,tr->Point1->z);
	//		glVertex3f(tr->Point2->x,tr->Point2->y,tr->Point2->z);
	//		glVertex3f(tr->Point3->x,tr->Point3->y,tr->Point3->z);
	//		glEnd();
	//	}


	//	glEnable(GL_LIGHTING);
	//	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
	//	GLfloat mat_color1[] = { 0.2, 0.6, 0.9 };
	//	GLfloat mat_shininess =  110.0f ;
	//	GLfloat specRefl[] = {1.0, 1.0, 1.0, 1.0f}; // default
	//	glBegin(GL_TRIANGLES);
	//	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mat_color1);
	//	glMaterialfv(GL_FRONT, GL_SPECULAR, specRefl); 
	//	glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
	//	glNormal3f(tr->Point1->getNormal(0),tr->Point1->getNormal(1),tr->Point1->getNormal(2));
	//	glVertex3f(tr->Point1->x,tr->Point1->y,tr->Point1->z);
	//	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mat_color1);
	//	glMaterialfv(GL_FRONT, GL_SPECULAR, specRefl); 
	//	glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
	//	glNormal3f(tr->Point2->getNormal(0),tr->Point2->getNormal(1),tr->Point2->getNormal(2));
	//	glVertex3f(tr->Point2->x,tr->Point2->y,tr->Point2->z);
	//	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mat_color1);
	//	glMaterialfv(GL_FRONT, GL_SPECULAR, specRefl); 
	//	glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
	//	glNormal3f(tr->Point3->getNormal(0),tr->Point3->getNormal(1),tr->Point3->getNormal(2));
	//	glVertex3f(tr->Point3->x,tr->Point3->y,tr->Point3->z);
	//	glEnd();
	//}
}



void WaterPlaneCUDA::drawTriangle(float3 p1,float3 p2,float3 p3,float3 n1,float3 n2,float3 n3){
	if (showEdges)
	{
		glDisable(GL_LIGHTING);
		glColor3f(1,1,1);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glBegin(GL_TRIANGLES);
		glVertex3f(p1.x,p1.y,p1.z);
		glVertex3f(p2.x,p2.y,p2.z);
		glVertex3f(p3.x,p3.y,p3.z);
		glEnd();
	}

	glEnable(GL_LIGHTING);
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
	GLfloat mat_color1[] = { 0.2, 0.6, 0.9 };
	GLfloat mat_shininess =  110.0f ;
	GLfloat specRefl[] = {1.0, 1.0, 1.0, 1.0f}; // default
	glBegin(GL_TRIANGLES);
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mat_color1);
	glMaterialfv(GL_FRONT, GL_SPECULAR, specRefl); 
	glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
	glNormal3f(n1.x,n1.y,n1.z);
	glVertex3f(p1.x,p1.y,p1.z);
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mat_color1);
	glMaterialfv(GL_FRONT, GL_SPECULAR, specRefl); 
	glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
	glNormal3f(n2.x,n2.y,n2.z);
	glVertex3f(p2.x,p2.y,p2.z);
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mat_color1);
	glMaterialfv(GL_FRONT, GL_SPECULAR, specRefl); 
	glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
	glNormal3f(n3.x,n3.y,n3.z);
	glVertex3f(p3.x,p3.y,p3.z);
	glEnd();


}

void WaterPlaneCUDA::initBuffer()
{

	//Start und Endkoordinaten für x-Richtung
	float startX = this->uLeft.x;
	float endX = this->lRight.x;

	//Start und Endkoordinaten für x-Richtung
	float startY = this->uLeft.z;
	float endY = this->lRight.z;

	int x = (int)(endX - startX)/stepSize;
	int y = (int)(endY - startY)/stepSize;

	CUDAvertices = new float3[pointsX*pointsY];
	CUDAnormals = new float3[pointsX*pointsY];
	//erzeuge entsprechend der Auflösung die benötigten Anzahl an Vertices und
	// die dazugehörigen Normalen
	int count = 0;
	for (float px = startX ; px< endX ; px+=stepSize){

		for (float py = startY; py < endY; py+=stepSize){
			
			CUDAvertices[count]=make_float3(px, this->baseHeight, py);
			CUDAnormals[count]=make_float3(0,1,0);
			count++;
			////Vertex
			//Vector *v = new Vector(px, this->baseHeight, py);
			//v->setNormal(0,1.0,0);
			//vertices.push_back(v);

		}
	}
}


WaterPlaneCUDA::~WaterPlaneCUDA(){
	cudaFree(gpu_normals);
	cudaFree(gpu_vertices);
	cudaFree(DIM);
}