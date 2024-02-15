#include "Tanh.h"

#include <cmath>

/********************************************** CONSTRUCTORS ************************************************/

Tanh::Tanh(unsigned IO_SZ_) : AbstractActivationFunction(IO_SZ_, "Tanh", true) {}


/********************************************** METHODS *****************************************************/

void Tanh::update_output(const VectorClass& input) {

    for (std::size_t i = 0; i < IO_SZ; ++i)
        output(i) = tanh( input(i) );

}


void Tanh::update_jacobian(const VectorClass& input){

    for (std::size_t i = 0; i < IO_SZ; ++i)
        jacobian(i,i) = 1 / ( cosh(input(i)) * cosh(input(i)) );

}
