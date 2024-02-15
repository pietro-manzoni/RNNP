/// Compute Mean Square Error
/**
 * Purpose-built \a metric class that computes a modified version of the "standard" MSE:
 * given a vector of input of prediction \a y_pred and a scalar actual value \a y_exact
 * of the monitored variable, returns
 * \f[
 *   ( y\_pred(0) - y\_exact )^2
 * \f]
 */

#ifndef MSE_metric_H
#define MSE_metric_H


#include "AbstractMetric.h"

class MSE_metric : public AbstractMetric {

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor of the object
    /**
     * Based on the constructor of #AbstractMetric
     * @param INPUT_SZ_ : size of the vectors expected in input
     */
    explicit MSE_metric(unsigned INPUT_SZ_);


    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& y_pred, NN::type y_exact) override;

};


#endif //MSE_metric_H
