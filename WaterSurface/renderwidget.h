#pragma once
#include <QtOpenGL/QGLWidget>
#include <QtGui/QMouseEvent>
#include <QtGui/QKeyEvent>
#include <QtCore/QTimer>
#include "waterplane.h"

class QPoint;

class RenderWidget : public QGLWidget
{
	Q_OBJECT
public:
	RenderWidget(QWidget * parent = 0);
	~RenderWidget(void);
	void changeWaveSettings(float waveSize, float waveIntensity);

protected:
	void initializeGL(); 
	void initializeLights();
	void paintGL(); 
	void resizeGL(int w, int h);
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void keyPressEvent (QKeyEvent * event);
	

protected slots:
		void render();

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
	float radius;	
	float theta;
	float phi;
	float fps;
	float surfaceSize;
	float waveSize;
	bool clicked;
	void drawAxis();
	void printw(float x, float y, float z, char* format, float fps);
	void drawFPS(void);
	void mouseDisturbSurface(QPoint);
	
};
