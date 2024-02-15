/// Compute Gaussian %CRPS with Lambda modification


#ifndef GAUSSIANCRPSTRIVIAL_H
#define GAUSSIANCRPSTRIVIAL_H


#include "AbstractLossFunction.h"
#include "../utilities.h"


class GaussianCRPSTrivial : public AbstractLossFunction{

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor of the object
    /**
     * Based on the constructor of #AbstractLossFunction
     * @param INPUT_SZ_ : size of the vectors expected in input
     */
    GaussianCRPSTrivial(unsigned INPUT_SZ_, NN::type LAMBDA_ = 1.0);

    const NN::type LAMBDA;

    // norminv of lambda/2
    const NN::type NL2;

    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& y_pred, NN::type y_exact) override;

    void update_gradient(const VectorClass& y_pred, NN::type y_exact) override;

};


#endif //GAUSSIANCRPSTRIVIAL_H
