/// Leaky Rectified %Linear Unit activation function
/**
 * Class derived from AbstractActivationFunction that implements the \a Leaky \a Rectified  \a %Linear \a Unit
 * activation function, a modification of the standard ReLU activation function. \n
 * The behaviour of the layer is thus partially determined by the constant A, that affects the output
 * values in presence of negative inputs, and is usually supposed to be very small. \n
 */

#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "AbstractActivationFunction.h"

class LeakyReLU : public AbstractActivationFunction {

    /********************************************** ATTRIBUTES **************************************************/

private:

    /// Constant coefficient
    /**
     * It affects the filtering behaviour of the activation function in its left branch.
     * Notice that
     * - in case A=1, the activation function coincides with the Linear one
     * - in case A=0, the activation function coincides with the standard ReLU
     */
    const NN::type A; // slope of the left branch


    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor
    /**
     * Based on the AbstractActivationFunction constructor.
     * @param IO_SZ_: input/output size of the layer
     * @param A_: shape coefficient, can be omitted (in this case default is 0.01)
     */
    explicit LeakyReLU(unsigned IO_SZ_, NN::type A_ = 0.01);


    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& input) override;
    void update_jacobian(const VectorClass& input) override;

};


#endif //LEAKYRELU_H
