#include "Softmax.h"
#include "../DataStructures/AlgebraicOperations.h"

#include <cmath>

/********************************************** CONSTRUCTORS ************************************************/

Softmax::Softmax (unsigned IO_SZ_) : AbstractActivationFunction(IO_SZ_, "Softmax", false) {}


/********************************************** METHODS *****************************************************/

void Softmax::update_output(const VectorClass& input){

    NN::type denom = 0;
    for (unsigned i = 0; i < IO_SZ; ++i)
        denom += exp( input(i) );

    for (std::size_t i = 0; i < IO_SZ; ++i)
        output(i) = exp( input(i) ) / denom;

}


void Softmax::update_jacobian(const VectorClass& input){

    NN::type denom = 0;
    for (unsigned i = 0; i < IO_SZ; ++i)
        denom += exp( input(i) );

    for (unsigned i = 0; i < IO_SZ; ++i) {
        const NN::type CACHE_OUTPUT_i = exp( input(i) ) / denom;
        const NN::type CACHE_SQUARED = CACHE_OUTPUT_i * CACHE_OUTPUT_i;
        for (unsigned j = 0; j < IO_SZ; ++j)
            jacobian(i, j) = - exp(input(j) - input(i)) * CACHE_SQUARED;
        jacobian(i, i) += CACHE_OUTPUT_i;
    }

}
