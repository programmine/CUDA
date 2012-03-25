#include "wmatrix.h"

#include <cassert>
#include <iostream>

//X - Achse: Columns
//Y - Achse: Rows
//Eement (0,0) ist "oben links"
WMatrix::WMatrix(unsigned int columns, unsigned int rows)
{
	this->columnCount = columns;

	this->rowCount = rows;

	values = new float*[columns];

	for (unsigned int i = 0; i< columns; i++){
		values[i] = new float[rows];
	}

	//Initialisiere mit 0.0
	for (unsigned int column = 0; column < columns; column++){
		for (unsigned int row = 0; row < rows; row++){
			values[column][row] = 0.0;
		}
	}
}

WMatrix::~WMatrix(void)
{
	delete[] values;
}

void WMatrix::setElement(unsigned int column, unsigned int row, float value)
{
	if ((row >= rowCount)||(row < 0)) return;
	if ((column >= columnCount)||(column < 0)) return;
	values[column][row] = value;
}

float WMatrix::getElement(unsigned int column, unsigned int row)
{
	return values[column][row];
}

int WMatrix::getColumnCount()
{
	return this->columnCount;
}

int WMatrix::getRowCount()
{
	return this->rowCount;
}

void WMatrix::printMatrix()
{

	for (unsigned int row = 0; row < rowCount; row++){

		for (unsigned int column = 0; column < columnCount; column++){

			std::cout<<"| "<<values[column][row];
		}
		std::cout << " |" << std::endl;
	}
}

int WMatrix::getSize()
{
	return columnCount * rowCount;
}

