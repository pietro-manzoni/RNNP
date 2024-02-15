#include "ExpLU.h"

#include <cmath>

/********************************************** CONSTRUCTORS ************************************************/

ExpLU::ExpLU(unsigned IO_SZ_, NN::type A_) : AbstractActivationFunction(IO_SZ_, "ExpLU", true), A(A_) {}


/********************************************** METHODS *****************************************************/

void ExpLU::update_output(const VectorClass& input){

    for (unsigned i = 0; i < IO_SZ; ++i) {
        if (input(i) >= 0)
            output(i) = input(i);
        else
            output(i) = A * ( exp(input(i)) - 1 );
    }

}


void ExpLU::update_jacobian(const VectorClass& input) {

    for (std::size_t i = 0; i < IO_SZ; ++i)
        if (input(i) >= 0)
            jacobian(i,i) = 1;
        else
            jacobian(i,i) = A * exp(input(i));

}
