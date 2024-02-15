/// Compute Gaussian %LogLikelihood
/**
 * Purpose-built \a loss \a function class that, given a vector of input of prediction \a y_pred and a
 * scalar actual value \a y_exact of the monitored variable, returns
 * \f[
 *   - 1000 * \log \mathcal{G}(y\_exact \vert y\_pred(0), y\_pred(1))
 * \f]
 * where \f$ \mathcal{G}(\cdot \vert \mu, \sigma) \f$ is the pdf of a Gaussian distribution
 * with mean \f$ \mu \f$ and standard deviation \f$ \sigma \f$.
 */


#ifndef LOGLIKELIHOOD_H
#define LOGLIKELIHOOD_H


#include "AbstractLossFunction.h"


class LogLikelihood : public AbstractLossFunction{

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor of the object
    /**
     * Based on the constructor of #AbstractLossFunction
     * @param INPUT_SZ_ : size of the vectors expected in input
     */
    explicit LogLikelihood(unsigned INPUT_SZ_);

    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& y_pred, NN::type y_exact) override;

    void update_gradient(const VectorClass& y_pred, NN::type y_exact) override;

};


#endif //LOGLIKELIHOOD_H
