/// Compute Gaussian %CRPS


#ifndef GAUSSIANCRPS_H
#define GAUSSIANCRPS_H


#include "AbstractLossFunction.h"
#include "../utilities.h"


class GaussianCRPS : public AbstractLossFunction{

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor of the object
    /**
     * Based on the constructor of #AbstractLossFunction
     * @param INPUT_SZ_ : size of the vectors expected in input
     */
    explicit GaussianCRPS(unsigned INPUT_SZ_);

    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& y_pred, NN::type y_exact) override;

    void update_gradient(const VectorClass& y_pred, NN::type y_exact) override;

};


#endif //GAUSSIANCRPS_H
