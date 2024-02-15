/// Hyperbolic Tangent activation function
/**
 * Class derived from AbstractActivationFunction that implements the \a %Tanh
 * activation function. \n
 * The activation function acts component-wise and is described by the following map:
 *  \f[
 *  \mathcal{A}(x_i) = \tanh(x_i) =
 *  \frac{ e^{x_i} - e^{-x_i} } { e^{x_i} + e^{-x_i} }
 *  \f]
 */

#ifndef TANH_H
#define TANH_H

#include "AbstractActivationFunction.h"

class Tanh : public AbstractActivationFunction{

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor
    /**
     * Based on the AbstractActivationFunction constructor.
     * @param IO_SZ_: input/output size of the layer
     */
    explicit Tanh(unsigned IO_SZ_);


    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& input) override;
    void update_jacobian(const VectorClass& input) override;

};


#endif //TANH_H
