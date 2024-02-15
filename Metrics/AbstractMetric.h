/// \a Pure \a abstract \a class that serves as a blueprint for any \a metric
/**
 * The present class is a fundamental part of a Neural Network since it allows to
 * evaluate the goodness of predicting power of the network itself. \n
 * The main distinction between #AbstractLossFunction and #AbstractMetric
 * is that the former has to provide a notion of gradient and suitable methods for
 * its update, while the latter is involved just in the forward propagation and thus
 * admits a simpler structure. \n
 * Thanks to inheritance, a LossFunction can be used as Metric (and gradient-related
 * methods are neglected).
 */

#ifndef ABSTRACTMETRIC_H
#define ABSTRACTMETRIC_H

#include "../global.h"

#include "../DataStructures/Matrix.h"
#include "../DataStructures/VectorClass.h"
#include "../DataStructures/AlgebraicOperations.h"

class AbstractMetric {

    /********************************************** ATTRIBUTES **************************************************/

public:

    /// Input size of the "layer".
    /**
     * For this kind of RNN, it coincides with #NN::AUTOREG_SZ
     */
    const unsigned INPUT_SZ;

    /// Stored accuracy
    NN::type output;


    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor of the object
    /**
     * @param INPUT_SZ_: size of the vector passed as input to the layer, i.e. the output of the
     *                   entire network. For this kind of RNN, it coincides with #NN::AUTOREG_SZ
     */
    explicit AbstractMetric(unsigned INPUT_SZ_);


    /********************************************** METHODS *****************************************************/

public:

    /// Update the #output attribute.
    /**
     * Pure virtual method to be overridden in the child classes.
     * @param y_pred: input vector provided to the layer (corresponding to the output of
     *      the previous one). Must have length equal to #INPUT_SZ.
     * @param y_exact: actual value of the observed variable, used to compute the loss
     */
    virtual void update_output(const VectorClass& y_pred, NN::type y_exact) = 0;

};


#endif //ABSTRACTMETRIC_H
