#ifndef RNN_P__LIBRARY_BPTT_H
#define RNN_P__LIBRARY_BPTT_H


#include "AbstractGradientAlgo.h"

class BPTT : public AbstractGradientAlgo {

    /********************************************** ATTRIBUTES *********************************************/

protected:

    /// Collectors for the previous states of the network
    /**
     *  They are std::vectors used to store the previous affine and hidden states
     */
    std::vector<VectorClass> affine_state_history, hidden_state_history, output_history;

    /******************************************** CONSTRUCTORS *******************************************/

public:

    /// Constructor of the object.
    /**
     * @param dl1_:         First Layer of the network
     * @param dl2_:         Second Layer of the network
     * @param SEQ_LEN_:     Length of the sequences that will be processed (used for allocating memory)
     */
    BPTT(FirstLayer& dl1_, SecondLayer& dl2_,
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

    /// Forward propagation, storing the variables of interest
    /**
     * @param x_sequence_train:     sequence of VectorClass object to be processed (exogenous inputs)
     * @param y_real:               target value
     */
    void forward_propagation(const std::vector<VectorClass>& x_sequence_train, const NN::type y_real);


    /// Backward propagation of gradient, as prescribed by BPTT
    /**
     * @param x_sequence_train:     sequence of VectorClass object to be processed (exogenous inputs)
     * @param y_real:               target value
     */
    void backward_propagation(const std::vector<VectorClass>& x_sequence_train, const NN::type y_real);


    /// Auxiliary function for implementing recursive exploration of the computational tree
    /**
     * @param x_sequence_train:     sequence of VectorClass object to be processed (exogenous inputs)
     * @param current_gradient:     gradient of the next ("next-in-time") element
     * @param TIME_LAG:             current time lag
     */
    bool recursive_backprop(const std::vector<VectorClass>& x_sequence_train,
                            const VectorClass& previous_gradient, const unsigned TIME_LAG);


    /// Extract the contribution of "TIME_LAG-th" element to the gradient
    /**
     * @param layer:                Considered Layer (FirstLayer reference can accept also a SecondLayer object...)
     * @param layer_input:          Inputs provided to the layer (exogenous in case of FirstLayer, hidden state if Second)
     * @param recurrent_history:    History of all previous recurrent states (i.e. of all produced outputs)
     * @param current_gradient:     Gradient of the next layer ("g_i" in the formulas)
     * @param TIME_LAG:             Current Time Lag with respect to final output (Sequence_Length)
     */
    void extract_contribution(FirstLayer& layer,
                              const std::vector<VectorClass>& layer_input,
                              const std::vector<VectorClass>& recurrent_history,
                              const VectorClass& current_gradient,
                              const unsigned TIME_LAG);

};


#endif //RNN_P__LIBRARY_BPTT_H
