#include <cmath>

#include "GaussianWinkler.h"

/********************************************** CONSTRUCTORS ************************************************/

GaussianWinkler::GaussianWinkler(unsigned INPUT_SZ_, NN::type LAMBDA_) : AbstractLossFunction(INPUT_SZ_, "GaussianWinkler"), LAMBDA(LAMBDA_) {}


/********************************************** METHODS *****************************************************/

void GaussianWinkler::update_output(const VectorClass&  y_pred, NN::type y_exact){

    const NN::type z_ = (y_exact - y_pred(0)) / std::fabs(y_pred(1));

    output = LAMBDA * 2 / sqrt(2*M_PI)
             - std::fabs(z_) * log(2)
             - integratelogQ(std::fabs(z_));
    output *= (2 * std::fabs(y_pred(1)));


}

void GaussianWinkler::update_gradient(const VectorClass& y_pred, NN::type y_exact) {


    const NN::type z_ = ( y_exact - y_pred(0) ) / std::fabs(y_pred(1));

    gradient(0) = 2 * ( log(2) + log(normcdf(-abs(z_))+1e-8) );
    if (z_ < 0)
        gradient(0) *= -1; //correction due to sign(z_)

    gradient(1) = + LAMBDA * 4 / sqrt(2*M_PI) - 2 * integratelogQ(abs(z_))
                  + 2 * abs(z_) * log(normcdf(-abs(z_))+1e-8);

    // correction due to fabs
    if (y_pred(1) < 0)
      gradient(1) *= -1;


}
