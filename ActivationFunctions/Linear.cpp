#include "Linear.h"

/********************************************** CONSTRUCTORS ************************************************/

Linear::Linear(unsigned IO_SZ_) : AbstractActivationFunction(IO_SZ_, "Linear", true)
{
    for (unsigned i = 0; i < IO_SZ; ++i)
        jacobian(i,i) = 1;
}


/********************************************** METHODS *****************************************************/

void Linear::update_output(const VectorClass& input) {
    output = input;
}


void Linear::update_jacobian(const VectorClass&){
    //in this case function is useless, since jacobian is constant
    return;
}
