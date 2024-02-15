#include <cmath>

#include "GaussianCRPSTrivial.h"

/********************************************** CONSTRUCTORS ************************************************/

GaussianCRPSTrivial::GaussianCRPSTrivial(unsigned INPUT_SZ_, NN::type LAMBDA_) :
  AbstractLossFunction(INPUT_SZ_, "GaussianCRPSTrivial"),
  LAMBDA(LAMBDA_),
  NL2( norminv(0.5*LAMBDA_) ) {}


/********************************************** METHODS *****************************************************/

void GaussianCRPSTrivial::update_output(const VectorClass& y_pred, NN::type y_exact){

    const NN::type z_ = (y_exact - y_pred(0)) / std::fabs(y_pred(1));

    output = (LAMBDA-1) * normpdf(NL2)
              - 1/sqrt(M_PI) * normcdf( sqrt(2)*NL2 )
              + normpdf(z_)
              + std::fabs(z_) * (0.5*LAMBDA - normcdf(-std::fabs(z_)));

    output *= (2 * std::fabs(y_pred(1)) / LAMBDA);

}


void GaussianCRPSTrivial::update_gradient(const VectorClass& y_pred, NN::type y_exact){

    const NN::type z_ = (y_exact - y_pred(0)) / std::fabs(y_pred(1));

    // Gaussian CRPS (come sopra) senza 2*sigma/LAMBDA
    const NN::type tmp_output =
                (LAMBDA-1) * normpdf(NL2)
              - 1/sqrt(M_PI) * normcdf( sqrt(2)*NL2 )
              + normpdf(z_)
              + std::fabs(z_) * (0.5*LAMBDA - normcdf(-std::fabs(z_)));


    gradient(0) = 2 / LAMBDA * normcdf(-std::fabs(z_)) - 1;
    if (z_ < 0)
        gradient(0) *= -1; //correction due to sign(z_)

    gradient(1) = tmp_output * 2 / LAMBDA + z_ * gradient(0);

    // correction due to fabs in definition of z_
    if (y_pred(1) < 0)
        gradient(1) *= -1;

}
