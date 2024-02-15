/// Compute Gaussian %CRPS with Lambda modification


#ifndef GAUSSIANCRPSLAMBDA_H
#define GAUSSIANCRPSLAMBDA_H


#include "AbstractLossFunction.h"
#include "../utilities.h"


class GaussianCRPSLambda : public AbstractLossFunction{

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor of the object
    /**
     * Based on the constructor of #AbstractLossFunction
     * @param INPUT_SZ_ : size of the vectors expected in input
     */
    GaussianCRPSLambda(unsigned INPUT_SZ_, NN::type LAMBDA_ = 1.0);

    const NN::type LAMBDA;

    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& y_pred, NN::type y_exact) override;

    void update_gradient(const VectorClass& y_pred, NN::type y_exact) override;

};


#endif //GAUSSIANCRPSLAMBDA_H
