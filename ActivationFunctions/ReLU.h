/// Rectified %Linear Unit activation function
/**
 * Class derived from AbstractActivationFunction that implements the \a Rectified  \a %Linear \a Unit
 * activation function. \n
 *  Because of the violent impact on the negative inputs, many modifications of this activation function
 *  have been developed. \n
 */

#ifndef RELU_H
#define RELU_H

#include "AbstractActivationFunction.h"

class ReLU : public AbstractActivationFunction{

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor
    /**
     * Based on the AbstractActivationFunction constructor.
     * @param IO_SZ_: input/output size of the layer
     */
    explicit ReLU(unsigned IO_SZ);


    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& input) override;
    void update_jacobian(const VectorClass& input) override;

};


#endif //RELU_H
