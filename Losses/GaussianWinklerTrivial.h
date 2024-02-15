/// Compute Gaussian Winkler Loss

#ifndef GAUSSIANWINKLERTRIVIAL_H
#define GAUSSIANWINKLERTRIVIAL_H


#include "AbstractLossFunction.h"
#include "../utilities.h"


class GaussianWinklerTrivial : public AbstractLossFunction{

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor of the object
    /**
     * Based on the constructor of #AbstractLossFunction
     * @param INPUT_SZ_ : size of the vectors expected in input
     */
    GaussianWinklerTrivial(unsigned INPUT_SZ_, NN::type LAMBDA_ = 1.0);


    const NN::type LAMBDA;

protected:

    // useful quantities

    // norminv of lambda/2
    const NN::type NL2;

    // integral of logQ from 0 to -NL2
    const NN::type INT_NL2;

    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& y_pred, NN::type y_exact) override;

    void update_gradient(const VectorClass& y_pred, NN::type y_exact) override;

};


#endif //GAUSSIANWINKLERTRIVIAL_H