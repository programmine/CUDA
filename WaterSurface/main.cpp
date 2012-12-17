#include "watersurface.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	if (argc < 3) return 0;
	
	int resLevel = atoi(argv[2]);
	bool cudaEnabled = atoi(argv[1]) == 1;
	WaterSurface w(cudaEnabled,resLevel);
	w.show();
	return a.exec();
}
