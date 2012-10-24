#ifndef WMATRIX_H
#define WMATRIX_H

/**
 * Helper class matrix.
 */
class WMatrix
{
public:

	/**
	 * constructor
	 */
	WMatrix(unsigned int columns, unsigned int rows);

	/**
	 * get value of the element with the given coordinates.
	 */
	virtual float getElement(int column, int row);

	/**
	 * set value of the element with the given coordinates.
	 */
	virtual void setElement(int column, int row, float value);

	/**
	 * size of the Matrix.
	 */
	virtual int getSize(void);

	/**
	 * row count
	 */
	virtual int getRowCount(void);

	/**
	 * column count
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
