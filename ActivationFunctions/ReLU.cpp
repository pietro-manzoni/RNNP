#include "ReLU.h"

/********************************************** CONSTRUCTORS ************************************************/

ReLU::ReLU(unsigned IO_SZ_) : AbstractActivationFunction(IO_SZ_, "ReLU", true) {}


/********************************************** METHODS *****************************************************/

void ReLU::update_output(const VectorClass& input) {

    for (unsigned i = 0; i < IO_SZ; ++i)
        output(i) = ( input(i) > 0 ) ? input(i) : 0;

}


void ReLU::update_jacobian(const VectorClass& input){

    for (std::size_t i = 0; i < IO_SZ; ++i)
        if (input(i) > 0)
            jacobian(i,i) = 1;
        else
            jacobian(i,i) = 0;

}
