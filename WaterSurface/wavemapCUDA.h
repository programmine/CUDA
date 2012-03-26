#ifndef WAVEMAPCUDA_H
#define WAVEMAPCUDA_H

class WMatrix;
#include "wavemap.h"

/**
 * Das Höhenfeld der WaterPlane.
 *
 * Zur Berechnung der Höhenfeldwerttveränderungen werden zwei 
 * Exemplare der Klasse: WMatrix benötgt um den DoubleBuffer - Mechanismus zu realisieren.
 * So kann der alte Zustand (Belegung der Höhenfelder) gehalten werden
 * um auf von diesem Zustand ausgehend die neue Belegung der Höhenfelder 
 * zu berechnen. Für den nächsten Schritt tauschen die Buffer die Rollen.
 */
class WaveMapCUDA : WaveMap
{
public:

	/**
	 * Berechnet die aktuelle Höhenfledbelegung
	 */
	void updateWaveMap();

};
#endif



