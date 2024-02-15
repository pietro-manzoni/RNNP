#include <cmath>

#include "LogLikelihood.h"

/********************************************** CONSTRUCTORS ************************************************/

LogLikelihood::LogLikelihood(unsigned INPUT_SZ_) : AbstractLossFunction(INPUT_SZ_, "LogLikelihood") {}


/********************************************** METHODS *****************************************************/

void LogLikelihood::update_output(const VectorClass& y_pred, NN::type y_exact){

    const NN::type normalized_tmp_ = (y_exact - y_pred(0)) / y_pred(1);

    // compute Gaussian Loglikelihood
    output = - .5 * log(2 * M_PI)
             - .5 * log( y_pred(1) * y_pred(1) )
             - .5 * normalized_tmp_ * normalized_tmp_;

    // compute negative Gaussian LogLikelihood (maximize)
    output *= -1;

}


void LogLikelihood::update_gradient(const VectorClass& y_pred, NN::type y_exact){

    const NN::type normalized_tmp_ = (y_exact - y_pred(0)) / y_pred(1);

    // compute gradient of Gaussian LogLikelihood ("pointing" in increase direction!)
    gradient(0) = normalized_tmp_ / y_pred(1);
    gradient(1) = (normalized_tmp_ * normalized_tmp_ - 1) / y_pred(1);

    // invert, because we use need negative Gaussian LogLikelihood
    gradient *= -1;

}
