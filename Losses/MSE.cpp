#include "MSE.h"

/********************************************** CONSTRUCTORS ************************************************/

MSE::MSE(unsigned INPUT_SZ_) : AbstractLossFunction(INPUT_SZ_, "MSE") {}


/********************************************** METHODS *****************************************************/

void MSE::update_output(const VectorClass&  y_pred, NN::type y_exact){
    output = (y_pred(0) - y_exact) * (y_pred(0) - y_exact);
}

void MSE::update_gradient(const VectorClass& y_pred, NN::type y_exact) {

    gradient(0) = 2 * (y_pred(0) - y_exact);

}
