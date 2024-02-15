/// %Softmax activation function
/**
 * Class derived from AbstractActivationFunction that implements the \a %Softmax
 * activation function. \n
 * The peculiarity of this activation function is that the i-th output is a function of all
 * the inputs of the layer (and not only the i-th, as usually happens). Therefore, the jacobian
 * matrix of the outputs with respect to the inputs is a dense matrix and not a diagonal one.
 * The activation function is represented by the following map:
 *  \f[
 *  \mathcal{A}(x_i) =
 *  \frac{ e^{x_i} } { \sum_j e^{x_j} }
 *  \f]
 */

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "AbstractActivationFunction.h"

class Softmax : public AbstractActivationFunction{


    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor
    /**
     * Based on the AbstractActivationFunction constructor.
     * @param IO_SZ_: input/output size of the layer
     */
    explicit Softmax(unsigned IO_SZ_);


    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& input) override;
    void update_jacobian(const VectorClass& input) override;

};


#endif //SOFTMAX_H
