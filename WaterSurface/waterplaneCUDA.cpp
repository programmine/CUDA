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

// Singleton
WaterPlane* WaterPlaneCUDA::getWaterPlane(){

	if(WaterPlaneExemplar == 0){
		WaterPlaneExemplar = new WaterPlaneCUDA;
	}
	return WaterPlaneExemplar;
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

	for (unsigned int x = 0; x< pointsX ; x++){

		for (unsigned int y = 0; y < pointsY; y++){

			vIndex = (y * pointsX) + x;
			Vector *v = vertices.at(vIndex);

			std::vector<Triangle*> neighbourTr = triangles->GetNeighbourTriangles(v);

			Vector *normal = new Vector(0,0,0);
			for (int triangleIndex = 0; triangleIndex < neighbourTr.size(); triangleIndex++)
			{
				Triangle *tr = neighbourTr.at(triangleIndex);
				*normal = *normal + *(tr->UpdateNormal());
			}
			Vector normalizedNormal = Vector::Normalize(*normal);
			v->setNormal(normalizedNormal.get(0),normalizedNormal.get(1),normalizedNormal.get(2));
		}
	}
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
