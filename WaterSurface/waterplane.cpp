#include "waterplane.h"
#include "wavemap.h"
#include "vector.h"
#include "trianglelist.h"
#include "triangle.h"
#include <cstdlib>
#include <iostream> 
#include <math.h>

WaterPlane* WaterPlane::WaterPlaneExemplar = 0;

WaterPlane::WaterPlane(){
	showEdges=false;
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


void WaterPlane::configure(Vector upperLeft, Vector lowerRight, float dampFactor, float resolution)
{

	stepSize = 1.0f/resolution;

	resolutionFactor = resolution;

	//reale Z - Achse ist x - Achse der WaterPlane
	sizeX = (int) abs(upperLeft.z - lowerRight.z);

	//reale X -Achse ist y- Achse der WaterPlane
	sizeY = (int) abs(upperLeft.x - lowerRight.x);

	//Anzahl der Netzpunkte in X -Richtung
	pointsX = (int)(sizeX * resolution);

	//Anzahl der Netzpunkte in Y -Richtung
	pointsY = (int)(sizeY * resolution);

	uLeft = upperLeft;

	lRight = lowerRight;

	//Der "Meeresspiegel"
	baseHeight = lRight.y;

	//triangles = new TriangleList();

	//Das Höhenfeld der Waterplane
	waveMap = new WaveMap(pointsX, pointsY, dampFactor);

	initBuffer();

	//setupTriangleDataStructure();

	drawMesh();
}

void WaterPlane::setupTriangleDataStructure()
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

void WaterPlane::createVBO(GLuint* vbo, int size)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//CUT_CHECK_ERROR_GL();
}


// create index buffer for rendering quad mesh
void WaterPlane::createMeshIndexBuffer(GLuint *id, int w, int h)
{
	int size = ((w*2)+2)*(h-1)*sizeof(GLuint);

	// create index buffer
	glGenBuffersARB(1, id);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

	// fill with indices for rendering mesh as triangle strips
	GLuint *indices = (GLuint *) glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
	if (!indices) {
		return;
	}

	for(int y=0; y<h-1; y++) {
		for(int x=0; x<w; x++) {
			*indices++ = y*w+x;
			*indices++ = (y+1)*w+x;
		}
		// start new strip with degenerate triangle
		*indices++ = (y+1)*w+(w-1);
		*indices++ = (y+1)*w;
	}

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}



void WaterPlane::initBuffer()
{
	//vertices.clear();

	//Start und Endkoordinaten für x-Richtung
	float startX = this->uLeft.x;
	float endX = this->lRight.x;

	//Start und Endkoordinaten für x-Richtung
	float startY = this->uLeft.z;
	float endY = this->lRight.z;

	vertices = new Vector[pointsX*pointsY];
	normals = new Vector[pointsX*pointsY];
	unsigned int count = 0;
	for (float px = startX ; px< endX ; px+=stepSize){
		for (float py = startY; py < endY; py+=stepSize){
			vertices[count].x = px;
			vertices[count].y = 0;
			vertices[count].z = py;
			normals[count].x = 0;
			normals[count].y = 1;
			normals[count].z = 0;
			count++;
		}
	}

	createVBO(&vertexBuffer, pointsX*pointsY*sizeof(float)*3);
	createVBO(&normalBuffer, pointsX*pointsY*sizeof(float)*3);
	createMeshIndexBuffer(&indexVertexBuffer, pointsX, pointsY);
	createMeshIndexBuffer(&indexNormalBuffer, pointsX, pointsY);

	glBindBuffer(GL_ARRAY_BUFFER, *&vertexBuffer);
	glBufferData( GL_ARRAY_BUFFER, pointsY*pointsX*3*sizeof(float), vertices, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, *&normalBuffer);
	glBufferData( GL_ARRAY_BUFFER, pointsY*pointsX*3*sizeof(float), normals, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);


	//erzeuge entsprechend der Auflösung die benötigten Anzahl an Vertices und
	// die dazugehörigen Normalen
	//for (float px = startX ; px< endX ; px+=stepSize){

	//	for (float py = startY; py < endY; py+=stepSize){

	//		//Vertex
	//		Vector *v = new Vector(px, this->baseHeight, py);
	//		v->setNormal(0,1.0,0);
	//		vertices.push_back(v);
	//	}
	//}
}


void WaterPlane::disturb(Vector disturbingPoint)
{
	float value = 0.5;

	float y = disturbingPoint.x;
	float x = disturbingPoint.z;
	float h = value;

	this->push(x,y,h);
}

void WaterPlane::disturbArea(float xmin, float zmin, float xmax, float zmax, float height)
{
	if ((zmin < this->uLeft.z) || (zmin > this->lRight.z)) return;
	if ((xmin < this->uLeft.x) || (xmin > this->lRight.x)) return;

	float radius = (float)(getWaterPlaneX(xmax)-getWaterPlaneX(xmin))/2.0f;
	if (radius <= 0) radius = 1;
	float centerX = (float)getWaterPlaneX((xmax+xmin)/2.0f);
	float centerZ = (float)getWaterPlaneY((zmax+zmin)/2.0f);
	float r2 = radius * radius;

	unsigned int xminW = getWaterPlaneX(xmin);
	unsigned int zminW = getWaterPlaneY(zmin);
	unsigned int xmaxW = getWaterPlaneX(xmax);
	unsigned int zmaxW = getWaterPlaneY(zmax);

	for(unsigned int x = xminW; x <= xmaxW; x++)
	{
		for (unsigned int y = zminW; y <= zmaxW; y++)
		{
			float insideCircle = ((x-centerX)*(x-centerX))+((y-centerZ)*(y-centerZ))-r2;
			
			if (insideCircle <= 0) waveMap->push((x),(y), (insideCircle/r2)*height);
		} 
	}
}


void WaterPlane::push(float x, float y, float depth)
{
	//Teste ob Punkt innerhalb der WaterPlane liegt:
	if (x > this->uLeft.z && x < this->lRight.z)
	{
		if (y > this->uLeft.x && y < this->lRight.x)
		{
			waveMap->push(getWaterPlaneX(x),getWaterPlaneY(y), depth);
		} else {
			std::cout<<"Point("<<x<<", "<<y<<") is not within waterplane"<<std::endl;
		}
	} else {
		std::cout<<"Point("<<x<<", "<<y<<") is not within waterplane"<<std::endl;
	}
}

int WaterPlane::getWaterPlaneX(float realX)
{
	int px = 0;

	if (realX >= 0){
		px = (int) ((realX * this->resolutionFactor)+0.5f);
	} else {
		px = (int) ((realX * this->resolutionFactor) + this->uLeft.z+0.5f);
	}
	return px;
}


int WaterPlane::getWaterPlaneY(float realY)
{
	int py = 0;

	if (realY >= 0){
		py = (int) (realY * this->resolutionFactor+0.5f);
	} else {
		py = (int) ((realY * this->resolutionFactor) + this->uLeft.x+0.5f);
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

	
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vertexBuffer);
	Vector* vertices2 = (Vector*)glMapBufferARB(GL_ARRAY_BUFFER_ARB, GL_WRITE_ONLY_ARB);

	for (int x = 0; x< pointsX ; x++){

		for (int y = 0; y < pointsY; y++){

			//neuer Höhenwert
			n = waveMap->getHeight(x,y);

			n += this->baseHeight; 

			vIndex = (y * pointsX) + x;

			Vector v = vertices2[vIndex];
			v.y = n;
			vertices2[vIndex] = v;
		}
	}
	glUnmapBufferARB(GL_ARRAY_BUFFER_ARB);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalBuffer);
	Vector* normals2 = (Vector*)glMapBufferARB(GL_ARRAY_BUFFER_ARB, GL_WRITE_ONLY_ARB);

	for (int x = 0; x< pointsX ; x++){

		for (int y = 0; y < pointsY; y++){

			
			/*
			   1  2
			*--*--*
			| /| /|
		  6 *--*--* 3
			| /| /|
			*--*--*
			5  4

			*/
			if (pointsX > 100){
				int a = 0;
			}
			nIndex = (y * pointsX) + x;
			Vector vertex = vertices2[nIndex];
			Vector newNormal = Vector();
			if (x+1 < pointsX){
				Vector v1 = vertices2[nIndex+1];
				Vector s1 = v1 - vertex;
				if ((y-1) >= 0){
					Vector v2 = vertices2[nIndex-pointsX+1];
					Vector s2 = v2 - vertex;
					newNormal = newNormal + s2.crossProduct(s1);
					Vector t = s2.crossProduct(s1);
				}
				
				if (y+1 < pointsY){
					Vector v6 = vertices2[nIndex+pointsX];
					Vector s6 = v6 - vertex;
					newNormal = newNormal + s1.crossProduct(s6);
				}
			}
			if (x-1 >= 0){
				Vector v4 = vertices2[nIndex-1];
				Vector s4 = v4 - vertex;
				if (y+1 < pointsY){
					Vector v5 = vertices2[nIndex+pointsX-1];
					Vector s5 = v5 - vertex;
					newNormal = newNormal + s5.crossProduct(s4);
				}

				if (y-1 >= 0){
					Vector v3 = vertices2[nIndex-pointsX];
					Vector s3 = v3 - vertex;
					newNormal = newNormal + s4.crossProduct(s3);
				}
			}
			normals2[nIndex] = Vector::Normalize(newNormal);
			
		}
		glUnmapBufferARB(GL_ARRAY_BUFFER_ARB);
	}
	
}


void WaterPlane::drawMesh()
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

void WaterPlane::deleteVBO(GLuint* vbo)
{
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}


//Destructor
WaterPlane::~WaterPlane(void){
	deleteVBO(&vertexBuffer);
	deleteVBO(&indexVertexBuffer);

	delete waveMap;
	
}

