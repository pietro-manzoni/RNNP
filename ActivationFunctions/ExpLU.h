/// Exponential %Linear Unit activation function
/**
 * Class derived from AbstractActivationFunction that implements the \a Exponential \a %Linear \a Unit
 * activation function. \n
 * The behaviour of the layer is thus partially determined by the constant A, that affects the output
 * values in presence of negative inputs. \n
 */


#ifndef EXPLU_H
#define EXPLU_H

#include "AbstractActivationFunction.h"

class ExpLU : public AbstractActivationFunction {

    /********************************************** ATTRIBUTES **************************************************/

private:

    /// Constant coefficient
    /**
     * It affects the filtering behaviour of the activation function in its left branch.
     * Notice that in case A=1, the function is \f$ \mathcal{C}^1 \f$ continuous.
     */
    const NN::type A;


    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor
    /**
     * Based on the AbstractActivationFunction constructor.
     * @param IO_SZ_: input/output size of the layer
     * @param A_: shape coefficient, can be omitted (in this case default is 1)
     */
    explicit ExpLU(unsigned IO_SZ_, NN::type A_ = 1);


    /********************************************** METHODS *****************************************************/

public:

    void update_output(const VectorClass& input) override;
    void update_jacobian(const VectorClass& input) override;

};


#endif //EXPLU_H