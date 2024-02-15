/// \a Pure \a abstract \a class that serves as a blueprint for any \a loss \a function.
/**
 * The present class is a fundamental part of a Neural Network since it allows to
 * evaluate the goodness of predicting power of the network itself. But not only,
 * since the Loss Function represents the starting point of the backpropagation
 * of the gradient. \n
 * Indeed, the main distinction between #AbstractLossFunction and #AbstractMetric
 * is that the former has to provide a notion of gradient and suitable methods for
 * its update, while the latter is involved just in the forward propagation and thus
 * admits a simpler structure. \n
 */

#ifndef ABSTRACTLOSSFUNCTION_H
#define ABSTRACTLOSSFUNCTION_H


#include "../global.h"

#include "../DataStructures/Matrix.h"
#include "../DataStructures/VectorClass.h"
#include "../DataStructures/AlgebraicOperations.h"


class AbstractLossFunction{

    /********************************************** ATTRIBUTES **************************************************/

public:

    /// Name of the Loss Function
    /**
     *  It is set during the construction of the object and cannot be modified later.
     */
    const std::string LOSS_NAME;

    /// Input size of the "layer".
    /**
     * For this kind of RNN, it coincides with #NN::AUTOREG_SZ
     */
    const unsigned INPUT_SZ;

    /// Stored accuracy
    NN::type output;

    /// Gradient of the output with respect to the inputs
    /**
     * Gradient of the (scalar!) output with respect to the inputs of the LossFunction, which in
     * turn are the output prediction of the entire network. It is thus a vector of size #NN::AUTOREG_SZ.
     * NB: gradient is supposed to be the direction of (positive!) growth. The optimizer automatically inverts
     * its sign to find the minimum.
     */
    VectorClass gradient;


    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor of the object
    /**
     * @param INPUT_SZ_:  size of the vector passed as input to the layer, i.e. the output of the
     *                    entire network. For this kind of RNN, it coincides with #NN::AUTOREG_SZ
     * @param LOSS_NAME_: name of the loss function
     */
    AbstractLossFunction(unsigned INPUT_SZ_, std::string LOSS_NAME_);


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


    /// Update the #gradient attribute.
    /**
     * Pure virtual method to be overridden in the child classes.
     * @param y_pred: input vector provided to the layer (corresponding to the output of
     *      the previous one). Must have length equal to #INPUT_SZ.
     * @param y_exact: actual value of the observed variable, used to compute the loss
     */
    virtual void update_gradient(const VectorClass& y_pred, NN::type y_exact) = 0;

};


#endif //ABSTRACTLOSSFUNCTION_H
