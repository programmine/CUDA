#define _USE_MATH_DEFINES

#include "renderwidget.h"
#include "waterplaneCUDA.h"
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <QtCore/QTime>
#include <QMouseEvent>
#include <QtCore/QPoint>
#include <QtOpenGL>
#include <iostream>
#include <math.h> 
#include <time.h>


RenderWidget::RenderWidget(bool cudaEnabled, int resolutionLevel)
: QGLWidget() 
{
	QGLFormat newFormat;
	newFormat.setSampleBuffers(true);
	newFormat.setDepth(true);

	setFormat(newFormat);
	setFocusPolicy(Qt::StrongFocus);
	width=1024;
	height=768;
	aspectRatio=width/height;
	resize(width,height);

	frameCount = 0;
	currentTime = 0;
	previousTime = 0;
	waveTimeCounter = 0;
	fps = 0.0f;

	radius = 5.0f;	
	theta  = (2.8f * M_PI) / 2.0f;
	phi = M_PI/1.2f;

	disturbAreaMin = 1.30f;
	disturbAreaMax = 1.60f;
	disturbHeight = 0.1f;

	// parameter controls resolution
	if (resolutionLevel == 1) resolution = 32.0f;
	else resolution = 64.0f;
	
	//and whether to use CUDA or not
	this->cudaEnabled = cudaEnabled;
	
	damping = 16;
	surfaceSize = 5.0;
	waveSize = 0.1;
	waterMode = 0;

	srand ( time(NULL) );

	clicked=false;

	setAutoFillBackground(false);
}

RenderWidget::~RenderWidget(void)
{
	makeCurrent(); 
	delete waterplane;
}

void RenderWidget::changeWaveSettings(float waveSize, float waveIntensity){
	this->waveSize=waveSize;
	this->disturbHeight=waveIntensity;
}


float RenderWidget::getFPS(void)
{
	frameCount++;
	currentTime = glutGet(GLUT_ELAPSED_TIME);

	//  Calculate time passed
	int timeInterval = currentTime - previousTime;

	if(timeInterval > 1000)
	{
		//  calculate the number of frames per second
		fps = frameCount / (timeInterval / 1000.0f);

		//  Set time
		previousTime = currentTime;

		//  Reset frame count
		frameCount = 0;
	}
	emit frameCounterChanged(fps);
	return timeInterval;
}

void RenderWidget::mouseDisturbSurface(QPoint lastPos)
{
	glLoadIdentity(); 
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT); 

	float Cx = radius * cos(theta) * sin(phi);
	float Cy = radius * cos(phi);
	float Cz = radius * sin(theta) * sin(phi);

	gluLookAt (Cx, -Cy, Cz, 0.0, 0, 0.0, 0, 1.0, 0.0);

	double dXfrom, dYfrom, dZfrom;
	double dXto, dYto, dZto;
	GLdouble modelMatrix[16];
	glGetDoublev(GL_MODELVIEW_MATRIX,modelMatrix);
	GLdouble projMatrix[16];
	glGetDoublev(GL_PROJECTION_MATRIX,projMatrix);

	int viewport[4];
	glGetIntegerv(GL_VIEWPORT,viewport);
	double dClickY = viewport[3] - double (lastPos.y());
	gluUnProject(
		lastPos.x(),
		dClickY,
		0,
		modelMatrix,
		projMatrix,
		viewport,
		&dXfrom,
		&dYfrom, 
		&dZfrom 
		);

	gluUnProject(
		lastPos.x(),
		dClickY,
		1,
		modelMatrix,
		projMatrix,
		viewport,

		&dXto, //-&gt; pointer to your own position (optional)
		&dYto, // id
		&dZto // id
		);
	double dYDir = dYto-dYfrom;
	double t = (-dYfrom) / dYDir;

	double dXDir = dXto-dXfrom;
	double dZDir = dZto-dZfrom;

	double surfacePointX = dXfrom + t*dXDir;
	double surfacePointY = 0;
	double surfacePointZ = dZfrom + t*dZDir;
	waterplane->disturbArea(surfacePointZ+(surfaceSize/2.0-waveSize),surfacePointX+(surfaceSize/2.0-waveSize),surfacePointZ+(surfaceSize/2.0+waveSize),surfacePointX+(surfaceSize/2.0+waveSize),disturbHeight);

}

void RenderWidget::mousePressEvent(QMouseEvent *event)
{
	lastPos = event->pos();

	if ((event->modifiers()) & Qt::ControlModifier) return;

	mouseDisturbSurface(lastPos);
}

void RenderWidget::resetWaterPlane()
{
	waterplane->configure(Vector(0,0,0),Vector(surfaceSize,0,surfaceSize),damping,resolution);
}

void RenderWidget::keyPressEvent (QKeyEvent * event)
{
	if (event->key()==Qt::Key_Space)
	{
		waterplane->update();
	}
	else if (event->key()==Qt::Key_R)
	{
		waterplane->configure(Vector(0,0,0),Vector(surfaceSize,0,surfaceSize),damping,resolution);
	}
	else if (event->key()==Qt::Key_E)
	{
		waterplane->toggleEdges();
	}

}

void RenderWidget::toggleEdges()
{
	waterplane->toggleEdges();
}

void RenderWidget::setWaterMode(int mode)
{
	waterMode = mode;
}

void RenderWidget::render()
{
	paintGL();
}

void RenderWidget::mouseMoveEvent(QMouseEvent *event)
{
	if ((event->modifiers()) & Qt::ControlModifier)
	{
		GLfloat dx = GLfloat(event->x() - lastPos.x()) / width;
		GLfloat dy = GLfloat(event->y() - lastPos.y()) / height;
		if (event->buttons() & Qt::LeftButton) {
			theta +=  (2*dx);
			phi +=  (2*dy);
			if(phi >= (M_PI/9) * 8) phi = (M_PI/9) * 8;
			if(phi <= (M_PI/9)) phi = (M_PI/9);

		} else if (event->buttons() & Qt::RightButton) {
			radius += (3*dy);
		}
		lastPos = event->pos();
	}else{
		mouseDisturbSurface(event->pos());
	}
}

void RenderWidget::resizeGL(int w, int h)
{
	width = w;
	height = h;
	aspectRatio = (float) width / (float) height;
	
	glViewport(0, 0, (GLsizei)width, (GLsizei)height); 
	glMatrixMode(GL_PROJECTION); 
	glLoadIdentity(); 
	gluPerspective(60, (GLfloat)width / (GLfloat)height, 1.0, 100.0);   
	glMatrixMode(GL_MODELVIEW);
	
}

void RenderWidget::applyWaterMode(){
	float xPos = 0;
	float zPos = 0;
	float curWaveTime = 0;
	switch(waterMode){
		case 0: break;
		default: 
			xPos = (rand() % 500 + 1) / 100.0f;
			zPos = (rand() % 500 + 1) / 100.0f;
			waterplane->disturbArea(xPos-waveSize/1.5f,zPos-waveSize/1.5f,xPos+waveSize/1.5f,zPos+waveSize/1.5f,disturbHeight/2.0f);
			break;
	}
}

void RenderWidget::paintGL()
{
	applyWaterMode();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glLoadIdentity(); 
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT); 

	float Cx = radius * cos(theta) * sin(phi);
	float Cy = radius * cos(phi);
	float Cz = radius * sin(theta) * sin(phi);
	
	gluLookAt (Cx, -Cy, Cz, 0.0, 0, 0.0, 0, 1.0, 0.0);
	glTranslatef(-surfaceSize/2.0,0,-surfaceSize/2.0);

	float timePassed = getFPS();
	waterplane->update();
	waterplane->drawMesh();
	
	glTranslatef(surfaceSize/2.0,0,surfaceSize/2.0);
	
	update();
}


void RenderWidget::initializeGL()
{
	glEnable(GL_DEPTH_TEST);
	initializeLights();
	glewInit();

	if (this->cudaEnabled) waterplane=WaterPlaneCUDA::getWaterPlane();
	else waterplane=WaterPlane::getWaterPlane();
	waterplane->configure(Vector(0,0,0),Vector(surfaceSize,0,surfaceSize),damping,resolution);
	waterplane->update();
}

void RenderWidget::initializeLights(){

	float lightX=-surfaceSize/2.0,lightY=surfaceSize/2.0f,lightZ=-surfaceSize/2.0;
	//ambient, specular and diffuse light
	GLfloat light_ambient[] = { 0.1, 0.1, 0.1, 1.0 };
	GLfloat light_specular[] = { 1, 1, 1, 1.0};
	GLfloat light_diffuse[] = { 0.6,0.6,0.6, 1.0 };
	GLfloat light_position[] = { lightX, lightY, lightZ, 1.0 };
	glClearColor (0.0, 0.0, 0.0, 0.0);
	glLightModeli( GL_LIGHT_MODEL_LOCAL_VIEWER, true);
	glLightfv(GL_LIGHT0,GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);

}
