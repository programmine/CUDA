#ifndef WATERPLANE_H
#define WATERPLANE_H


class WaveMap;
class WMatrix;
class TriangleList;
#include "vector.h"
#include <vector>

/**
 * Die Wasseroberflaeche.
 *
 * Erzeugt realistische Wellenbewegung und Spiegelung der Umgebung.
 * Die WaterPlane ist ein Singleton.
 *
 */
class WaterPlane
{
public:
	/**
	 * Erzeugt den naechsten Wellenzustand.
	 */
	void update(void);

	/**
	 * Gibt die WaterPlane zurck.
	 */
	static WaterPlane* getWaterPlane();

	/**
	 * Zeichnet das Mesh der Wasseroberflaeche.
	 */
	void drawMesh(void);

	/**
	 *
	 * Initialisiert die WaterPlane. 
	 * 
	 * @param dampFactor Ausdehnung der erzeugten Wellen
	 * @param resolution beschreibt wieviel Gitternetzpunkte pro Laengeneinheit erzeugt werden
	 * 
	 */
	void configure(Vector upperLeft, Vector lowerRight, float dampFactor, float resolution);

	/**
	 * Stoert die WaterPlane.
	 *
	 * @param disturbingPoint Punkt an dem die Waterplane gestoert werden soll.
	 * 
	 */
	void disturb(Vector disturbingPoint);

	void disturbArea(float xmin, float zmin, float xmax, float zmax, float height);

	/**
	 * Stoert die WaterPlane am spezifizierten Punkt.
	 */
	void disturb(float x, float y, float z);

	/**
	 * Fgt einen Punkt zu der Liste der automatisch gestoerten Punkte hinzu, 
	 * um ein realistischen Effekt zu erzielen werden auch die direkten Nachbarn 
	 * des Punktes gestoert.Fgt einen Punkt hinzu, der von der WaterPlane automatisch gestoert werden soll.
	 */
	void appendDisturbedPoint(Vector point);

	void clear();

	void toggleEdges();

	void toggleNormals();

	~WaterPlane(void);

private:

	/**
	 * Standard Konstruktor
	 */
	WaterPlane();

	/**
	 * Das WaterPlane Exemplar
	 */
	static WaterPlane* WaterPlaneExemplar;

	/** 
	 * Vertauscht die Rollen der beiden Buffer fuer den alten und neuen Wellenzustand.
	 */
	void swapBuffers(void);

	/**
	 * Setzt den Hoehenwert an Stelle (x,y) auf depth.
	 */
	void push(float x, float y, float depth);

	/**
	 * Initialisiert die benoetigten Datenstrukturen.
	 */
	void initBuffer(void);

	/**
	 * Zeichnet die Environment - Map.
	 */
	void drawEnvironmentMap(void);

	/**
	 * Die Waterplpane stoert sich selbst an definierten Punkten 
	 * um eine staendige Wasserbewegung zu simulieren.
	 */
	void motion(void);

	/**
	 * Die Hoehe des Punktes an der Stelle (x,y).
	 */
	int getHeight(int x, int y);

	/**
	 * Errechnet zu einer X - Koordinate aus dem Weltkoordinatensystem 
	 * den zugehoerigen Punkt in der Wasseroberflaeche.
	 */
	unsigned int getWaterPlaneX(float realX);

	/**
	 * Errechnet zu einer Y - Koordinate aus dem Weltkoordinatensystem 
	 * den zugehoerigen Punkt in der Wasseroberflaeche.
	 */
	unsigned int getWaterPlaneY(float realY);

	void setupDataStructure(void);

	float resolutionFactor;

	//Vertices der Wasseroberflaeche
	std::vector<Vector*> vertices;

	//Die Normalen zu den Vertices
	std::vector<Vector*> vNormals;

	//Liste der automatisch gestoerten Punkte
	std::vector<Vector*> disturbedPoints;
	/** 
	 * Diese Datenstruktuer speichert die Hoehenwerte zum aktuellen Zustand
	 */
	WaveMap* waveMap;

	bool doMotion;

	double startTime;

	Vector uLeft;

	Vector lRight;

	unsigned int pointsX;

	unsigned int pointsY;

	unsigned int sizeX;

	unsigned int sizeY;

	unsigned int currentDisturbedPoint;

	float stepSize;

	float baseHeight;

	double elapsedTime;

	TriangleList *triangles;

	bool showEdges;

	bool calcNormals;
};

#endif
