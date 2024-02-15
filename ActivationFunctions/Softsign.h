/// %Softsign activation function
/**
 * Class derived from AbstractActivationFunction that implements the \a %Softsign
 * activation function. \n
 * The activation function acts component-wise and is described by the following map:
 *  \f[
 *  \mathcal{A}(x_i) =
 *  \frac{ x_i } { 1 + \vert x_i \vert }
 *  \f]
 */

#ifndef SOFTSIGN_H
#define SOFTSIGN_H

#include "AbstractActivationFunction.h"

class Softsign : public AbstractActivationFunction{

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor
    /**
     * Based on the AbstractActivationFunction constructor.
     * @param IO_SZ_: input/output size of the layer
     */
    explicit Softsign(unsigned IO_SZ_);


    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& input) override;
    void update_jacobian(const VectorClass& input) override;

};


#endif //SOFTSIGN_H
