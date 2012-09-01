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
	virtual float getElement(int column, int row);

	/**
	 * Setzt den Wert des jeweiligen Elements zurück.
	 */
	virtual void setElement(int column, int row, float value);

	/**
	 * Die Größe der Matrix.
	 */
	virtual int getSize(void);

	/**
	 * Anzahl der Zeilen.
	 */
	virtual int getRowCount(void);

	/**
	 * Anzahl der Spalten.
	 */
	virtual int getColumnCount(void);

	virtual void printMatrix(void);

	virtual ~WMatrix(void);

	float* Values;

private:

	unsigned int rowCount;
	unsigned int columnCount;

	
};

#endif
