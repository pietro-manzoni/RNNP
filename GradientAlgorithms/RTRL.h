//
// Created by Pietro Manzoni on 17/02/22.
//

#ifndef RNN_P__LIBRARY_RTRL_H
#define RNN_P__LIBRARY_RTRL_H

#include "AbstractGradientAlgo.h"

class RTRL : public AbstractGradientAlgo {

    /***************************************** ATTRIBUTES *****************************************/

protected:

    /// Jacobian matrix of the output of each layer wrt weigths
    /**
     *  These are the main building blocks for RTRL.
     */
    Matrix dl1_jacobian_params, dl2_jacobian_params;

    /// Collectors for the previous Jacobian matrices
    /**
     *  They are Circular Buffers with length equal to the maximum time lag
     */
    CircularBuffer<Matrix> dtheta_buffer, dphi_buffer;


    /**************************************** CONSTRUCTORS ****************************************/

public:

    /// Constructor of the object.
    /**
     * @param dl1_:         First Layer of the network
     * @param dl2_:         Second Layer of the network
     * @param SEQ_LEN_:     Length of the sequences that will be processed (used for allocating memory)
     */
    RTRL(FirstLayer& dl1_, SecondLayer& dl2_,
         unsigned SEQUENCE_LENGTH_);


    /***************************************** METHODS ************************************************/

public:

    /// Process a sequence, computing output (forward propagation) and gradient of loss (backward propagation)
    /**
     * @param x_sequence_train:     sequence of VectorClass object to be processed (exogenous inputs)
     * @param y_real:               target value
     */
    void process_sequence(const std::vector<VectorClass>& x_sequence_train, const NN::type y_real) override;


protected:


    void dl1_partial_derivative(const VectorClass& x_sequence, const unsigned T, const Matrix& mtrx);

    void dl2_partial_derivative(void);

};


#endif //RNN_P__LIBRARY_RTRL_H
