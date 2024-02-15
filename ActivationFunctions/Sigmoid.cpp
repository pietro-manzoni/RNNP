#include "Sigmoid.h"

#include <cmath>

/********************************************** CONSTRUCTORS ************************************************/

Sigmoid::Sigmoid(unsigned IO_SZ_) : AbstractActivationFunction(IO_SZ_, "Sigmoid", true) {}


/********************************************** METHODS *****************************************************/

void Sigmoid::update_output(const VectorClass& input) {

    for (std::size_t i = 0; i < IO_SZ; ++i)
        output(i) = 1 / (   1 + exp( - input(i) )  );

}


void Sigmoid::update_jacobian(const VectorClass& input){

    for (std::size_t i = 0; i < IO_SZ; ++i) {
        const NN::type CACHE_SIGMOID_i = 1 / (   1 + exp( - input(i) )  );
        jacobian(i, i) = CACHE_SIGMOID_i * (1 - CACHE_SIGMOID_i);
    }

}
