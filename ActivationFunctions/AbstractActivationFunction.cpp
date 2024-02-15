#include "AbstractActivationFunction.h"

#include <utility>


/********************************************** CONSTRUCTORS ************************************************/

AbstractActivationFunction::AbstractActivationFunction(unsigned IO_SZ_, std::string AF_NAME_, bool DIAGONAL_) :
        IO_SZ(IO_SZ_), AF_NAME(std::move(AF_NAME_)), DIAGONAL(DIAGONAL_), output( VectorClass(IO_SZ_, 0) ),
        jacobian( Matrix(IO_SZ_, IO_SZ_, 0) ) {}
