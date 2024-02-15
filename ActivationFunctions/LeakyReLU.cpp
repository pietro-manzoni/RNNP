#include "LeakyReLU.h"

/********************************************** CONSTRUCTORS ************************************************/

LeakyReLU::LeakyReLU(unsigned IO_SZ_, NN::type A_) : AbstractActivationFunction(IO_SZ_, "LeakyReLU", true), A(A_) {}


/********************************************** METHODS *****************************************************/

void LeakyReLU::update_output(const VectorClass& input){

    for (unsigned i = 0; i < IO_SZ; ++i) {
        if (input(i) >= 0)
            output(i) = input(i);
        else
            output(i) = A * input(i);
    }

}

void LeakyReLU::update_jacobian(const VectorClass& input) {

    for (std::size_t i = 0; i < IO_SZ; ++i)
        if (input(i) >= 0)
            jacobian(i,i) = 1;
        else
            jacobian(i,i) = A;

}
