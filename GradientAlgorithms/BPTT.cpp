
#include "BPTT.h"


BPTT::BPTT(FirstLayer& dl1_, SecondLayer& dl2_,
            unsigned SEQUENCE_LENGTH_) :

           AbstractGradientAlgo("BPTT", dl1_, dl2_, SEQUENCE_LENGTH_),

           //instantiate vectors for storing history
           affine_state_history(SEQUENCE_LENGTH, dl1.output),
           hidden_state_history(SEQUENCE_LENGTH, af1->output),
           output_history(SEQUENCE_LENGTH, dl2.output) {}


void BPTT::process_sequence(const std::vector<VectorClass>& x_sequence_train, const NN::type y_real){

    // perform forward propagation and store variables of interest
    forward_propagation(x_sequence_train, y_real);

    // perform gradient computation according to TRRL
    backward_propagation(x_sequence_train, y_real);

}


void BPTT::forward_propagation(const std::vector<VectorClass>& x_sequence_train, const NN::type y_real){

    // obtain prediction for the sequence x_sequence_train
    for (unsigned t = 0; t < SEQUENCE_LENGTH; ++t) {

        // forward propagation chain
        dl1.update_output(x_sequence_train[t], output_history, t);       // First Layer
        af1->update_output(dl1.output);	            			          // First Activation Function
        dl2.update_output(af1->output, output_history, t);	              // Second Layer

        //update stored values
        affine_state_history[t] = dl1.output;
        hidden_state_history[t] = af1->output;
        output_history[t] = dl2.output;

    }

    // NB:  All the outputs are assessed through the loss (for all t=1:SEQUENCE_LENGTH)
    //      However only the last element of a sequence is used for computing gradient (many-to-one architecture)
    loss->update_output(dl2.output, y_real);

}

void BPTT::backward_propagation( const std::vector<VectorClass>& x_sequence_train, const NN::type y_real){

    // reset gradients
    dl1.gradient_params.reset();
    dl2.gradient_params.reset();

    // backward propagation of input/output Jacobian matrices
    loss->update_gradient(output_history[SEQUENCE_LENGTH-1], y_real);

    //recursive backpropagation function
    recursive_backprop(x_sequence_train, loss->gradient, 0);

}


bool BPTT::recursive_backprop(const std::vector<VectorClass>& x_sequence_train,
                              const VectorClass& previous_gradient, const unsigned TIME_LAG){

    // base case:
    if (TIME_LAG >= SEQUENCE_LENGTH) {
        return false;
    }

    else {

        // compute phi gradient, using hidden_state_history and (useless!) output_history
        extract_contribution(dl2, hidden_state_history, output_history, previous_gradient, TIME_LAG);

        // compute jacobian of activation function
        af1->update_jacobian(affine_state_history[SEQUENCE_LENGTH-TIME_LAG-1]);

        // instantiate and compute current_gradient
        //const VectorClass current_gradient = (previous_gradient * dl2.kernel) * af1->jacobian;
        VectorClass current_gradient(af1->IO_SZ);
        if (af1->DIAGONAL)
            mult_vD( (previous_gradient * dl2.kernel), af1->jacobian, current_gradient);
        else
            mult_in_place( (previous_gradient * dl2.kernel), af1->jacobian, current_gradient);

        // compute theta gradient, using x_sequences_train and all_outputs
        extract_contribution(dl1, x_sequence_train, output_history, current_gradient, TIME_LAG);

        // recursive loop over the lags
        for (unsigned i = 0; i < dl1.N_LAGS; ++i){
            bool goahead = recursive_backprop( x_sequence_train,current_gradient * dl1.AR_kernels[i],TIME_LAG + dl1.LAGS[i]);

            // tree is over, not necessary to go ahead with other lags
            if (!goahead)
                break;
        }

    }

    return true;

}


void BPTT::extract_contribution(FirstLayer& layer,
                                const std::vector<VectorClass>& layer_input,
                                const std::vector<VectorClass>& recurrent_history,
                                const VectorClass& current_gradient,
                                const unsigned TIME_LAG){

    // kernel
    for (unsigned j = 0; j < layer.OUTPUT_SZ; ++j) {
        // base index (j rows have already been processed)
        const unsigned BASE = j * layer.INPUT_SZ;
        const unsigned ARRAY_INDEX = SEQUENCE_LENGTH - TIME_LAG - 1;  // index in the history array
        for (unsigned k = 0; k < layer.INPUT_SZ; ++k)
            layer.gradient_params(BASE + k) += current_gradient(j) * layer_input[ARRAY_INDEX].at(k);
    }

    // AR kernels
    for (unsigned l = 0; l < layer.N_LAGS; ++l) {
        // base index (some kernels have already been processed)
        const unsigned BASE_KERNELS = (layer.INPUT_SZ + l * NN::OUTPUT_NODES) * layer.OUTPUT_SZ;
        if (TIME_LAG + layer.LAGS[l] + 1 <= SEQUENCE_LENGTH) {  // do not consider dependencies wrt y(t), if t<0
            const unsigned ARRAY_INDEX = SEQUENCE_LENGTH - TIME_LAG - layer.LAGS[l] - 1;
            for (unsigned j = 0; j < layer.OUTPUT_SZ; ++j) {
                // base index (for this kernel, j rows have already been processed)
                const unsigned BASE = j * NN::OUTPUT_NODES;
                for (unsigned k = 0; k < NN::OUTPUT_NODES; ++k)
                    layer.gradient_params(BASE + BASE_KERNELS + k) += current_gradient(j) * recurrent_history[ARRAY_INDEX].at(k);
            }
        }
    }

    // bias
    // base index for bias vector (all kernels have already been processed)
    const unsigned BASE_BIAS = (layer.INPUT_SZ + layer.N_LAGS * NN::OUTPUT_NODES) * layer.OUTPUT_SZ;
    for (unsigned j = 0; j < layer.OUTPUT_SZ; ++j) {
        layer.gradient_params(BASE_BIAS + j) += current_gradient(j);
    }

}
