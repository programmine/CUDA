#pragma once
#include "waterplane.h"
#include <QtOpenGL/QGLWidget>
#include <QtGui/QMouseEvent>
#include <QtGui/QKeyEvent>
#include <QtCore/QTimer>


class QPoint;

// widget to contain the opengl widget
class RenderWidget : public QGLWidget
{
	Q_OBJECT
public:
	RenderWidget(QWidget * parent = 0);
	~RenderWidget(void);
	// functions called by watersurface.cpp from GUI
	void changeWaveSettings(float waveSize, float waveIntensity);
	void resetWaterPlane();
	void toggleEdges();
	void setWaterMode(int mode);

protected:
	//OpenGL functions
	void initializeGL(); 
	void initializeLights();
	//draws the whole scene
	void paintGL(); 
	void resizeGL(int w, int h);
	
	//keyboard and mouse events
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void keyPressEvent (QKeyEvent * event);
	

protected slots:
		void render();

signals:
		void frameCounterChanged(float fps);


private:
	int width;
	int height;
	float aspectRatio;
	QPoint lastPos;
	QTimer* animationTimer;
	WaterPlane *waterplane;
	float disturbAreaMin;
	float disturbAreaMax;
	float disturbHeight;
	float resolution;
	float damping;
	int frameCount;
	int currentTime;
	int previousTime;
	int waveTimeCounter;
	int prevWaveTime;
	float radius;	
	float theta;
	float phi;
	float fps;
	float surfaceSize;
	float waveSize;
	bool clicked;
	int waterMode;
	void drawAxis();
	void printw(float x, float y, float z, char* format, float fps);
	float getFPS(void);
	void mouseDisturbSurface(QPoint);
	void applyWaterMode();
	
};
