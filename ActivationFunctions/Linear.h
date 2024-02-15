/// %Linear (i.e. identity) activation function
/**
 * Class derived from AbstractActivationFunction that implements the \a %Linear
 * activation function. \n
 * The activation function acts component-wise and is described by the following map:
 *  \f[
 *  \mathcal{A}(x_i) = x_i
 *  \f]
 */

#ifndef LINEAR_H
#define LINEAR_H


#include "AbstractActivationFunction.h"

class Linear : public AbstractActivationFunction{

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor
    /**
     * Based on the AbstractActivationFunction constructor.
     * @param IO_SZ_: input/output size of the layer
     */
    explicit Linear(unsigned IO_SZ_);


    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& input) override;
    void update_jacobian(const VectorClass&) override;

};


#endif //LINEAR_H
