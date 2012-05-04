#ifndef WMATRIX_H
#define WMATRIX_H

/**
 * Behelfsklasse für Matrizen belieber Größe.
 */
class WMatrix
{
public:

	/**
	 * Der Konstruktor
	 */
	WMatrix(unsigned int columns, unsigned int rows);

	/**
	 * Gibt den Wert des jeweiligen Elements zurück.
	 */
	float getElement(unsigned int column, unsigned int row);

	/**
	 * Setzt den Wert des jeweiligen Elements zurück.
	 */
	void setElement(unsigned int column, unsigned int row, float value);

	/**
	 * Die Größe der Matrix.
	 */
	int getSize(void);

	/**
	 * Anzahl der Zeilen.
	 */
	int getRowCount(void);

	/**
	 * Anzahl der Spalten.
	 */
	int getColumnCount(void);

	void printMatrix(void);

public:
	~WMatrix(void);
	float** Values;

private:
	unsigned int rowCount;
	unsigned int columnCount;

	
};

#endif
