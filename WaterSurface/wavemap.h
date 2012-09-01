#ifndef WAVEMAP_H
#define WAVEMAP_H

class WMatrix;

/**
 * Das Höhenfeld der WaterPlane.
 *
 * Zur Berechnung der Höhenfeldwerttveränderungen werden zwei 
 * Exemplare der Klasse: WMatrix benötgt um den DoubleBuffer - Mechanismus zu realisieren.
 * So kann der alte Zustand (Belegung der Höhenfelder) gehalten werden
 * um auf von diesem Zustand ausgehend die neue Belegung der Höhenfelder 
 * zu berechnen. Für den nächsten Schritt tauschen die Buffer die Rollen.
 */
class WaveMap
{
public:

	/**
	 * Erzeugt eine neue WaveMap mit der Ausdehnung pX * pY und einem damp - Factor.
	 * Dieser wird für die Berechnung der Welle bzw. deren Ausdehnung herangezogen.
	 */
	WaveMap(int pX, int pY, float damp);
	WaveMap();

	/**
	 * Vertauscht die beiden Buffer in ihrer Rolle.
	 */
	void swap();

	/**
	 * Verhändert den Höhenwert um an der Stelle (x,y).
	 */
	void push(int x, int y, float value);

	/**
	 * Berechnet die aktuelle Höhenfledbelegung
	 */
	virtual void updateWaveMap();

	/**
	 * Anzahl der Punkte in X - Richtung
	 */
	int getPointsX();

	/**
	 * Anzahl der Punkte in Y - Richtung
	 */
	int getPointsY();

	/** 
	 * Höhenwert an Stelle (x,y)
	 */
	virtual float getHeight(int x, int y);

	virtual ~WaveMap(void);

protected:
	WMatrix* newWave;
	WMatrix* tmp;
	WMatrix* oldWave;
	
protected:
	int pointsX;
	int pointsY;
	
	float sizeX;
	float sizeY;

	float stepSize;
	
	float dampFactor;

};
#endif



