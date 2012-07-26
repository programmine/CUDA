#include "wavemapCUDA.h"
#include "vector.h"
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


void WaterPlaneCUDA::update()
{
	//update WaveMap
	waveMap->updateWaveMap();
	int vIndex = 0;
	float n = 0.0;
	for (int x = 0; x< pointsX ; x++){

		for (int y = 0; y < pointsY; y++){

			//neuer Höhenwert
			n = waveMap->getHeight(x,y);

			n += this->baseHeight; 

			vIndex = (y * pointsX) + x;

			float3 v = cpu_vertices[vIndex];
			v.y = n;
			cpu_vertices[vIndex] = v;
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, *&vertexBuffer);
	glBufferData( GL_ARRAY_BUFFER, pointsY*pointsX*3*sizeof(float), cpu_vertices, GL_DYNAMIC_DRAW);

	cutilSafeCall(cudaMemcpy(gpu_vertices,cpu_vertices,(pointsX*pointsX)*sizeof(float3),cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(gpu_normals,cpu_normals,(pointsX*pointsX)*sizeof(float3),cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(DIM,&pointsX,sizeof(unsigned int),cudaMemcpyHostToDevice));
	updateNormalsGPU1(gpu_vertices,gpu_normals,DIM);
	//cutilSafeCall(cudaMemcpy(cpu_vertices,gpu_vertices,(pointsX*pointsX)*sizeof(float3),cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(cpu_normals,gpu_normals,(pointsX*pointsX)*sizeof(float3),cudaMemcpyDeviceToHost));

	
	

	glBindBuffer(GL_ARRAY_BUFFER, *&normalBuffer);
	glBufferData( GL_ARRAY_BUFFER, pointsY*pointsX*3*sizeof(float), cpu_normals, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);


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

	glBindBuffer( GL_ARRAY_BUFFER, vertexBuffer );
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

	cpu_vertices = new float3[pointsX*pointsY];
	cpu_normals = new float3[pointsX*pointsY];
	unsigned int count = 0;
	for (float px = startX ; px< endX ; px+=stepSize){
		for (float py = startY; py < endY; py+=stepSize){
			cpu_vertices[count].x = px;
			cpu_vertices[count].y = 0;
			cpu_vertices[count].z = py;
			cpu_normals[count].x = 0;
			cpu_normals[count].y = 1;
			cpu_normals[count].z = 0;
			count++;
		}
	}

	createVBO(&vertexBuffer, pointsX*pointsY*sizeof(float)*3);
	createVBO(&normalBuffer, pointsX*pointsY*sizeof(float)*3);
	createMeshIndexBuffer(&indexVertexBuffer, pointsX, pointsY);
	createMeshIndexBuffer(&indexNormalBuffer, pointsX, pointsY);

	glBindBuffer(GL_ARRAY_BUFFER, *&vertexBuffer);
	glBufferData( GL_ARRAY_BUFFER, pointsY*pointsX*3*sizeof(float), cpu_vertices, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, *&normalBuffer);
	glBufferData( GL_ARRAY_BUFFER, pointsY*pointsX*3*sizeof(float), cpu_normals, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


WaterPlaneCUDA::~WaterPlaneCUDA(){
	cudaFree(gpu_normals);
	cudaFree(gpu_vertices);
	cudaFree(DIM);
}