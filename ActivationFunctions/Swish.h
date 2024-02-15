/// %Swish activation function
/**
 * Class derived from AbstractActivationFunction that implements the \a %Swish
 * activation function, a modification of the Sigmoid activation function.\n
 * The activation function acts component-wise and is described by the following map:
 *  \f[
 *  \mathcal{A}(x_i) =
 *  \frac{ x_i } { 1 + e^{-x_i} }
 *  \f]
 */

#ifndef SWISH_H
#define SWISH_H

#include "AbstractActivationFunction.h"

class Swish : public AbstractActivationFunction {

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor
    /**
     * Based on the AbstractActivationFunction constructor.
     * @param IO_SZ_: input/output size of the layer
     */
    explicit Swish(unsigned IO_SZ_);


    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& input) override;
    void update_jacobian(const VectorClass& input) override;

};


#endif //SWISH_H
