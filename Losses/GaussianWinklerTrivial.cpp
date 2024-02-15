#include <cmath>

#include "GaussianWinklerTrivial.h"

/********************************************** CONSTRUCTORS ************************************************/

GaussianWinklerTrivial::GaussianWinklerTrivial(unsigned INPUT_SZ_, NN::type LAMBDA_) :
    AbstractLossFunction(INPUT_SZ_, "GaussianWinklerTrivial"),
        LAMBDA(LAMBDA_),
        NL2( norminv(0.5*LAMBDA_) ),
        INT_NL2(  integratelogQ(-norminv(0.5*LAMBDA_))  ) {}

/********************************************** METHODS *****************************************************/

void GaussianWinklerTrivial::update_output(const VectorClass&  y_pred, NN::type y_exact){

    const NN::type z_ = (y_exact - y_pred(0)) / std::fabs(y_pred(1));

    output =  2 * normpdf(NL2) + std::fabs(z_) * log(0.5*LAMBDA)
            + NL2 * log(0.5*LAMBDA)
            - integratelogQ(std::fabs(z_)) + INT_NL2;

    output *= (2 * std::fabs(y_pred(1)));

}


void GaussianWinklerTrivial::update_gradient(const VectorClass& y_pred, NN::type y_exact) {


    const NN::type z_ = ( y_exact - y_pred(0) ) / std::fabs(y_pred(1));

    gradient(0) = 2 * ( log(normcdf(-std::fabs(z_))+1e-8) - log(0.5*LAMBDA) );
    if (z_ < 0)
        gradient(0) *= -1; //correction due to sign(z_)

    // Gaussian CWS (come sopra) senza 2*sigma

    const NN::type tmp_output = 2 * normpdf(NL2) + std::fabs(z_) * log(0.5*LAMBDA)
			       + NL2 * log(0.5*LAMBDA)
			       - integratelogQ(std::fabs(z_)) + INT_NL2;

    gradient(1) = 2 * tmp_output + z_ * gradient(0);

    // correction due to fabs
    if (y_pred(1) < 0)
      gradient(1) *= -1;

}
