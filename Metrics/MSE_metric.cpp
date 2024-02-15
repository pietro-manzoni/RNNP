
#include "MSE_metric.h"

/********************************************** CONSTRUCTORS ************************************************/

MSE_metric::MSE_metric(unsigned INPUT_SZ_) : AbstractMetric(INPUT_SZ_) {}

/********************************************** METHODS *****************************************************/

void MSE_metric::update_output(const VectorClass&  y_pred, NN::type y_exact){
    output = (y_pred(0) - y_exact) * (y_pred(0) - y_exact);
}

