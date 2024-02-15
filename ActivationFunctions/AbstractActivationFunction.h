/// \a Pure \a abstract \a class that serves as a blueprint for any \a activation \a function.
/**
 * Each activation function must override two methods with the indicated signature:
 * - void update_output(const VectorRn& input)
 * - void update_jacobian(const VectorRn& input)
 */


#ifndef ABSTRACTACTIVATIONFUNCTION_H
#define ABSTRACTACTIVATIONFUNCTION_H

#include "../global.h"

#include "../DataStructures/VectorClass.h"
#include "../DataStructures/Matrix.h"

class AbstractActivationFunction {

    /********************************************** ATTRIBUTES **************************************************/

public:

    /// Name of the activation function (e.g. Sigmoid, ReLU, ...)
    /**
     *  It is set during the construction of the object and cannot be modified later. Used for log creation.
     */
    const std::string AF_NAME;

    /// Input and output size of the activation function
    /**
     * It is set during the construction of the object and cannot be modified later.
     */
    const unsigned IO_SZ;


    /// Is the jacobian matrix diagonal?
    /**
     * True if the action of the activation function is component-wise and thus the jacobian
     * matrix is diagonal. It is a feature of the activation function and so it cannot be modified.
     */
    const bool DIAGONAL;

    /// Vector containing the output of the layer.
    /**
     * Has length #IO_SZ.
     */
    VectorClass output;

    /// Jacobian matrix
    /**
     * %Matrix containing the jacobian of the output of the activation function with respect to
     * the input thereof. Has fixed size: #IO_SZ x #IO_SZ
     */
    Matrix jacobian;


    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor of the object
    /**
     * @param IO_SZ_:       input/output size of the layer
     * @param AF_NAME_:       name of chosen activation function
     * @param DIAGONAL_:   is jacobian matrix diagonal?
     */
    AbstractActivationFunction(unsigned IO_SZ_, std::string AF_NAME_, bool DIAGONAL_);


    /********************************************** METHODS *****************************************************/

public:

    /// Update the #output attribute
    /**
     *  @param input: input vector provided to the layer (corresponding to the output of
     *      the previous one). Must have length equal to #IO_SZ.
     */
    virtual void update_output(const VectorClass& input) = 0;

    /// Update the #jacobian attribute
    /**
     *  @param input: input vector provided to the layer (corresponding to the output of
     *      the previous one). Must have length equal to #IO_SZ.
     */
    virtual void update_jacobian(const VectorClass& input) = 0;

};


#endif //ABSTRACTACTIVATIONFUNCTION_H
