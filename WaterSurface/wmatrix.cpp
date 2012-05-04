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

	Values = new float*[columns];

	for (unsigned int i = 0; i< columns; i++){
		Values[i] = new float[rows];
	}

	//Initialisiere mit 0.0
	for (unsigned int column = 0; column < columns; column++){
		for (unsigned int row = 0; row < rows; row++){
			Values[column][row] = 0.0;
		}
	}
}

WMatrix::~WMatrix(void)
{
	delete Values;
}

void WMatrix::setElement(unsigned int column, unsigned int row, float value)
{
	if ((row >= rowCount)||(row < 0)) return;
	if ((column >= columnCount)||(column < 0)) return;
	Values[column][row] = value;
}

float WMatrix::getElement(unsigned int column, unsigned int row)
{
	return Values[column][row];
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

			std::cout<<"| "<<Values[column][row];
		}
		std::cout << " |" << std::endl;
	}
}

int WMatrix::getSize()
{
	return columnCount * rowCount;
}

