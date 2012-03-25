#include "waterplane.h"
#include "wavemap.h"
#include "vector.h"
#include "trianglelist.h"
#include "triangle.h"
#include <windows.h>
#include <GL/gl.h>
#include <cstdlib>
#include <iostream> 
#include <math.h>

WaterPlane* WaterPlane::WaterPlaneExemplar = 0;

WaterPlane::WaterPlane(){
	elapsedTime = 0.0;
	showEdges=false;
	calcNormals=true;
}

// Singleton
WaterPlane* WaterPlane::getWaterPlane(){

	if(WaterPlaneExemplar == 0){
		WaterPlaneExemplar = new WaterPlane;
	}
	return WaterPlaneExemplar;
}

void WaterPlane::toggleEdges(){

	showEdges=!showEdges;
}

void WaterPlane::toggleNormals(){

	calcNormals=!calcNormals;
	if (calcNormals) return;
	int vIndex=0;
	for (unsigned int x = 0; x< pointsX ; x++){

		for (unsigned int y = 0; y < pointsY; y++){

			vIndex = (y * pointsX) + x;
			Vector *v = vertices.at(vIndex);
			v->setNormal(0,1,0);
		}
	}
}


void WaterPlane::configure(Vector upperLeft, Vector lowerRight, float dampFactor, float resolution)
{
	doMotion = false;

	stepSize = 1.0/resolution;

	resolutionFactor = resolution;

	//reale Z - Achse ist x - Achse der WaterPlane
	sizeX = (unsigned int) abs(upperLeft.get(2) - lowerRight.get(2));

	//reale X -Achse ist y- Achse der WaterPlane
	sizeY = (unsigned int) abs(upperLeft.get(0) - lowerRight.get(0));

	//Anzahl der Netzpunkte in X -Richtung
	pointsX = (unsigned int)(sizeX * resolution);

	//Anzahl der Netzpunkte in Y -Richtung
	pointsY = (unsigned int)(sizeY * resolution);

	uLeft = upperLeft;

	lRight = lowerRight;

	//Der "Meeresspiegel"
	baseHeight = lRight.get(1);

	triangles = new TriangleList();

	//Das Höhenfeld der Waterplane
	waveMap = new WaveMap(pointsX, pointsY, dampFactor);

	initBuffer();

	setupDataStructure();

	drawEnvironmentMap();

	drawMesh();

	currentDisturbedPoint = 0;

	startTime = 0;
}

void WaterPlane::clear()
{
	initBuffer();
}

void WaterPlane::setupDataStructure()
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


void WaterPlane::initBuffer()
{
	vertices.clear();
	vNormals.clear();

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

	//Liste für die Punkte die von der Wasseroberfläche automatisch gestört werden sollen
	disturbedPoints.clear();
}

void WaterPlane::appendDisturbedPoint(Vector point)
{
	float x = point.get(0);
	float y = point.get(1);
	float z = point.get(2);

	disturbedPoints.push_back(new Vector(x-1.0, y, z-1.0));
	disturbedPoints.push_back(new Vector(x+1.0, y, z-1.0));
	disturbedPoints.push_back(new Vector(x-1.0, y, z+1.0));
	disturbedPoints.push_back(new Vector(x+1.0, y, z+1.0));

	disturbedPoints.push_back(new Vector(x, y, z));

	disturbedPoints.push_back(new Vector(x-1.0, y, z));
	disturbedPoints.push_back(new Vector(x+1.0, y, z));
	disturbedPoints.push_back(new Vector(x, y, z+1.0));
	disturbedPoints.push_back(new Vector(x, y, z+1.0));
}

void WaterPlane::disturb(Vector disturbingPoint)
{
	float value = 0.5;

	float y = disturbingPoint.get(0);
	float x = disturbingPoint.get(2);
	float h = value;

	this->push(x,y,h);
}

void WaterPlane::disturbArea(float xmin, float zmin, float xmax, float zmax, float height)
{
	if ((zmin < this->uLeft.get(2)) || (zmin > this->lRight.get(2))) return;
	if ((xmin < this->uLeft.get(0)) || (xmin > this->lRight.get(0))) return;

	float radius = (getWaterPlaneX(xmax)-getWaterPlaneX(xmin))/2.0f;
	float centerX = getWaterPlaneX((xmax+xmin)/2.0f);
	float centerZ = getWaterPlaneY((zmax+zmin)/2.0f);
	float r2 = radius * radius;

	float xminW = getWaterPlaneX(xmin);
	float zminW = getWaterPlaneY(zmin);
	float xmaxW = getWaterPlaneX(xmax);
	float zmaxW = getWaterPlaneY(zmax);

	for(unsigned int x = xminW; x <= xmaxW; x++)
	{
		for (unsigned int y = zminW; y <= zmaxW; y++)
		{
			float insideCircle = ((x-centerX)*(x-centerX))+((y-centerZ)*(y-centerZ))-r2;
			if (insideCircle <= 0) waveMap->push((x),(y), (insideCircle/r2)*height);
		} 
	}
}


void WaterPlane::disturb(float x, float y, float z)
{
	float value = y;

	float py = x;
	float px = z;
	float h = value;

	this->push(px, py, h);
}

void WaterPlane::push(float x, float y, float depth)
{
	//Teste ob Punkt innerhalb der WaterPlane liegt:
	if (x > this->uLeft.get(2) && x < this->lRight.get(2))
	{
		if (y > this->uLeft.get(0) && y < this->lRight.get(0))
		{
			waveMap->push(getWaterPlaneX(x),getWaterPlaneY(y), depth);
		} else {
			std::cout<<"Point("<<x<<", "<<y<<") is not within waterplane"<<std::endl;
		}
	} else {
		std::cout<<"Point("<<x<<", "<<y<<") is not within waterplane"<<std::endl;
	}
}

unsigned int WaterPlane::getWaterPlaneX(float realX)
{
	unsigned int px = 0;

	if (realX >= 0){
		px = (unsigned int) ((realX * this->resolutionFactor)+0.5f);
	} else {
		px = (unsigned int) ((realX * this->resolutionFactor) + this->uLeft.get(2)+0.5f);
	}
	return px;
}


unsigned int WaterPlane::getWaterPlaneY(float realY)
{
	unsigned int py = 0;

	if (realY >= 0){
		py = (unsigned int) (realY * this->resolutionFactor);
	} else {
		py = (unsigned int) ((realY * this->resolutionFactor) + this->uLeft.get(0));
	}
	return py;
}


void WaterPlane::update()
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

	if (calcNormals)
	{
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
}


void WaterPlane::drawMesh()
{
	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	for (int triangleIndex = 0; triangleIndex < triangles->GetCount(); triangleIndex++)
	{
		Triangle *tr = triangles->GetTriangle(triangleIndex);
		if (showEdges)
		{
			glDisable(GL_LIGHTING);
			glColor3f(1,1,1);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glBegin(GL_TRIANGLES);
				glVertex3f(tr->Point1->get(0),tr->Point1->get(1),tr->Point1->get(2));
				glVertex3f(tr->Point2->get(0),tr->Point2->get(1),tr->Point2->get(2));
				glVertex3f(tr->Point3->get(0),tr->Point3->get(1),tr->Point3->get(2));
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
			glNormal3f(tr->Point1->getNormal(0),tr->Point1->getNormal(1),tr->Point1->getNormal(2));
			glVertex3f(tr->Point1->get(0),tr->Point1->get(1),tr->Point1->get(2));
			glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mat_color1);
			glMaterialfv(GL_FRONT, GL_SPECULAR, specRefl); 
			glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
			glNormal3f(tr->Point2->getNormal(0),tr->Point2->getNormal(1),tr->Point2->getNormal(2));
			glVertex3f(tr->Point2->get(0),tr->Point2->get(1),tr->Point2->get(2));
			glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, mat_color1);
			glMaterialfv(GL_FRONT, GL_SPECULAR, specRefl); 
			glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
			glNormal3f(tr->Point3->getNormal(0),tr->Point3->getNormal(1),tr->Point3->getNormal(2));
			glVertex3f(tr->Point3->get(0),tr->Point3->get(1),tr->Point3->get(2));
		glEnd();
	}
}

void WaterPlane::drawEnvironmentMap()
{
	
}


void WaterPlane::motion(){
}


//Destructor
WaterPlane::~WaterPlane(void){}

