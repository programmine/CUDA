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
	RenderWidget(bool cudaEnabled, int resolutionLevel);
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
	bool cudaEnabled;
	/// returns current fps and emits signal for ui to display fps
	float getFPS(void);
	/// called to disturb surface
	/// given 2D coordinate is transformed into 3D position and projected on water plane
	void mouseDisturbSurface(QPoint);
	/// function used to apply water mode (normal or rain)
	void applyWaterMode();
	QTimer *timer;
	
};
