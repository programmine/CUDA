#ifndef WATERSURFACE_H
#define WATERSURFACE_H

#include <QtGui/QMainWindow>
#include <QtGui/QWidget>
#include <QtGui/QComboBox>
#include "renderwidget.h"

class QDoubleSpinBox;
class QLabel;

class WaterSurface : public QMainWindow
{
	Q_OBJECT

public:
	WaterSurface(bool cudaEnabled, int resolutionLevel);
	~WaterSurface();



protected slots:
	void changeWaveSettingsSize(double);
	void changeWaveSettingsIntens(double);
	void resetWaveSettings();
	void toggleEdges(int);
	void setFrameCounter(float fps);
	void WaterModeChanges(int);

private:
	RenderWidget *renderWidget; 
	QWidget *centralWidget;
	QDoubleSpinBox *waveIntens;
	QDoubleSpinBox *waveSize;
	QLabel *framecounter;
	QComboBox *mode;
};

#endif // WATERSURFACE_H
