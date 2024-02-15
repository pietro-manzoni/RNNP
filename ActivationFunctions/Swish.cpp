#include "Swish.h"

#include <cmath>

/********************************************** CONSTRUCTORS ************************************************/

Swish::Swish(unsigned IO_SZ_) : AbstractActivationFunction(IO_SZ_, "Swish", true) {}


/********************************************** METHODS *****************************************************/

void Swish::update_output(const VectorClass& input) {

    for (std::size_t i = 0; i < IO_SZ; ++i)
        output(i) = input(i) / (   1 + exp( - input(i) )  );

}


void Swish::update_jacobian(const VectorClass& input){

    for (std::size_t i = 0; i < IO_SZ; ++i) {
        const NN::type CACHE_SIGMOID = 1 / (1 + exp(-input(i)));
        jacobian(i, i) = CACHE_SIGMOID + (1 - CACHE_SIGMOID) * CACHE_SIGMOID * input(i);
    }

}

