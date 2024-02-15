#include "Softsign.h"

#include <cmath>

/********************************************** CONSTRUCTORS ************************************************/

Softsign::Softsign(unsigned IO_SZ_) : AbstractActivationFunction(IO_SZ_, "Softsign", true) {}


/********************************************** METHODS *****************************************************/

void Softsign::update_output(const VectorClass& input){

    for (unsigned i = 0; i < IO_SZ; ++i)
        output(i) = input(i) / ( 1+fabs(input(i)) );

}

void Softsign::update_jacobian(const VectorClass& input){

    for (std::size_t i = 0; i < IO_SZ; ++i)
        jacobian(i,i) = 1  / ( (1+fabs(input(i))) * (1+fabs(input(i))) );

}
