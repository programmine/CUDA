#include "watersurface.h"
#include <QtGui/QVBoxLayout>
#include <QtGui/QGridLayout>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtOpenGL/QGLFormat>
#include <QtGui/QDoubleSpinBox>
#include <iostream>

WaterSurface::WaterSurface(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	renderWidget = new RenderWidget();
	centralWidget = new QWidget();
	QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
	QGridLayout *controlLayout = new QGridLayout();
	controlLayout->setAlignment(Qt::AlignLeft);
	controlLayout->addWidget(new QLabel("Wave Size"),1,1);
	waveSize = new QDoubleSpinBox();
	waveSize->setRange(0.01,0.5);
	waveSize->setSingleStep(0.01);
	waveSize->setFixedWidth(100);
	waveSize->setValue(0.1);
	controlLayout->addWidget(waveSize,1,2);

	controlLayout->addWidget(new QLabel("Wave Intensity"),1,3);
	waveIntens = new QDoubleSpinBox();
	waveIntens->setRange(0.05,0.5);
	waveIntens->setSingleStep(0.01);
	waveIntens->setValue(0.1);
	waveIntens->setFixedWidth(100);
	controlLayout->addWidget(waveIntens,1,4);

	QPushButton *apply = new QPushButton("Apply");
	apply->setFixedWidth(100);
	QObject::connect(apply, SIGNAL(clicked()),this, SLOT(changeWaveSettings()));
	QPushButton *reset = new QPushButton("Reset");
	reset->setFixedWidth(100);

	controlLayout->addWidget(apply,2,1);
	controlLayout->addWidget(reset,2,2);
	setCentralWidget(centralWidget);
	mainLayout->addLayout(controlLayout);
	mainLayout->addWidget(renderWidget);  
	setWindowTitle(tr("Water Surface"));
	resize(1044,800);

	std::cout<<"************Key Control************"<<std::endl;
	std::cout<<"*    E  -  toggle wireframe       *"<<std::endl;
	std::cout<<"*    R  -  reset water surface    *"<<std::endl;
	std::cout<<"*    B  -  random waves           *"<<std::endl;
	std::cout<<"*    N  -  toggle normals         *"<<std::endl;
	std::cout<<"* Ctrl+Mouse  -  navigation       *"<<std::endl;
	std::cout<<"***********************************"<<std::endl;
}

void WaterSurface::changeWaveSettings()
{
	float waveS = waveSize->value();
	float waveI = waveIntens->value();
	renderWidget->changeWaveSettings(waveS,waveI);
}

WaterSurface::~WaterSurface()
{

}
