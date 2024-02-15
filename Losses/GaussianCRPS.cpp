#include <cmath>

#include "GaussianCRPS.h"

/********************************************** CONSTRUCTORS ************************************************/

GaussianCRPS::GaussianCRPS(unsigned INPUT_SZ_) : AbstractLossFunction(INPUT_SZ_, "GaussianCRPS") {}


/********************************************** METHODS *****************************************************/

void GaussianCRPS::update_output(const VectorClass& y_pred, NN::type y_exact){

    const NN::type normalized_tmp_ = (y_exact - y_pred(0)) / y_pred(1);

    output = ( normalized_tmp_ * ( normcdf(normalized_tmp_) - 0.5 ) +
               normpdf(normalized_tmp_) - 1 / (2*sqrt(M_PI)) ) * y_pred(1);

}


void GaussianCRPS::update_gradient(const VectorClass& y_pred, NN::type y_exact){

    const NN::type normalized_tmp_ = (y_exact - y_pred(0)) / y_pred(1);

    gradient(0) = - ( normcdf(normalized_tmp_) - 0.5 );
    gradient(1) = normpdf(normalized_tmp_) - 1/(2*sqrt(M_PI));

}
