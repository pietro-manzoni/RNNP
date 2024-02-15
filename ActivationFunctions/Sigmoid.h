/// %Sigmoid activation function
/**
 * Class derived from AbstractActivationFunction that implements the \a %Sigmoid
 * activation function. \n
 * The activation function acts component-wise and is described by the following map:
 *  \f[
 *  \mathcal{A}(x_i) =
 *  \frac{1}{ 1 + e^{-x_i} }
 *  \f]
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "AbstractActivationFunction.h"

class Sigmoid : public AbstractActivationFunction{

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor
    /**
     * Based on the AbstractActivationFunction constructor.
     * @param IO_SZ_: input/output size of the layer
     */
    explicit Sigmoid(unsigned IO_SZ_);


    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& input) override;
    void update_jacobian(const VectorClass& input) override;

};


#endif //SIGMOID_H
