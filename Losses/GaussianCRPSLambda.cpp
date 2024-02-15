#include <cmath>

#include "GaussianCRPSLambda.h"

/********************************************** CONSTRUCTORS ************************************************/

GaussianCRPSLambda::GaussianCRPSLambda(unsigned INPUT_SZ_, NN::type LAMBDA_) : AbstractLossFunction(INPUT_SZ_, "GaussianCRPSLambda"), LAMBDA(LAMBDA_) {}


/********************************************** METHODS *****************************************************/

void GaussianCRPSLambda::update_output(const VectorClass& y_pred, NN::type y_exact){

    const NN::type z_ = (y_exact - y_pred(0)) / y_pred(1);

    output = ( ( normcdf(z_) - 0.5 ) * z_ + normpdf(z_) +
               (LAMBDA * (1-1/sqrt(2)) - 1) / sqrt(2*M_PI) ) * 2 * y_pred(1);

}


void GaussianCRPSLambda::update_gradient(const VectorClass& y_pred, NN::type y_exact){

    const NN::type z_ = (y_exact - y_pred(0)) / y_pred(1);

    gradient(0) = - 2 * ( normcdf(z_) - 0.5 );
    gradient(1) =   2 * ( normpdf(z_) + (LAMBDA * (1-1/sqrt(2)) - 1) / sqrt(2*M_PI) );

}
