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

	Values = new float[columns*rows];

	//Initialisiere mit 0.0
	for (unsigned int i = 0; i < (columns*rows); i++){
		Values[i] = 0.0;
	}
}

WMatrix::~WMatrix(void)
{
	delete Values;
}

void WMatrix::setElement(int column, int row, float value)
{
	if ((row >= rowCount)||(row < 0)) return;
	if ((column >= columnCount)||(column < 0)) return;
	int index = columnCount*column + row;
	Values[index] = value;
}

float WMatrix::getElement(int column, int row)
{
	int index = columnCount*column + row;
	return Values[index];
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
		}
		std::cout << " |" << std::endl;
	}
}

int WMatrix::getSize()
{
	return columnCount * rowCount;
}

