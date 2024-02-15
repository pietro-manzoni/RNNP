
#include "TRRL.h"

TRRL::TRRL(FirstLayer& dl1_, SecondLayer& dl2_,
           unsigned SEQUENCE_LENGTH_) :

           AbstractGradientAlgo("TRRL", dl1_, dl2_, SEQUENCE_LENGTH_),

            //instantiate vectors for storing history
           affine_state_history(SEQUENCE_LENGTH, dl1.output),
           hidden_state_history(SEQUENCE_LENGTH, dl1.output),
           output_history(SEQUENCE_LENGTH, dl2.output) {}


void TRRL::process_sequence(const std::vector<VectorClass>& x_sequence_train, const NN::type y_real){

    // perform forward propagation and store variables of interest
    forward_propagation(x_sequence_train, y_real);

    // perform gradient computation according to TRRL
    backward_propagation(x_sequence_train, y_real);

}

void TRRL::forward_propagation(const std::vector<VectorClass>& x_sequence_train, const NN::type y_real){


    // process a sequence
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

    // NB:  All the outputs are assessed through the loss and the metric (for all t=1:SEQUENCE_LENGTH)
    //      However only the last element of a sequence is used for computing gradient (many-to-one)
    loss->update_output(dl2.output, y_real);

}

void TRRL::backward_propagation(const std::vector<VectorClass>& x_sequence_train, const NN::type y_real){

    // reset gradients
    dl1.gradient_params.reset();
    dl2.gradient_params.reset();

    // backward propagation of input/output Jacobian matrices
    loss->update_gradient(output_history[SEQUENCE_LENGTH-1], y_real);

    // auxiliary vectors: the former considers whether the gradient g_i is empty or not, the second collects all g_i
    std::vector<VectorClass> gradients(SEQUENCE_LENGTH, VectorClass(NN::OUTPUT_NODES, 0));

    // fill first element
    gradients[0] = loss->gradient;

    for (unsigned time_lag = 0; time_lag < SEQUENCE_LENGTH; ++time_lag) {

        // add contribution to phi gradient, using hidden_state_history and (useless!) output_history
        extract_contribution(dl2, hidden_state_history, output_history,
                             gradients[time_lag],time_lag);

        // compute jacobian of activation function
        af1->update_jacobian(affine_state_history[SEQUENCE_LENGTH - time_lag - 1]);

        // instantiate and compute temporary g_i
        //const VectorClass temporary_gradient = (gradients[time_lag] * dl2.kernel) * af1->jacobian;
        VectorClass temporary_gradient(af1->IO_SZ);
        if (af1->DIAGONAL)
            mult_vD( (gradients[time_lag] * dl2.kernel), af1->jacobian, temporary_gradient);
        else
            mult_in_place( (gradients[time_lag] * dl2.kernel), af1->jacobian, temporary_gradient);

        // add contribution to theta gradient, using x_sequence_train and all_outputs
        extract_contribution(dl1, x_sequence_train, output_history,
                             temporary_gradient, time_lag);

        // update g_i gradients
        for (unsigned i = 0; i < dl1.N_LAGS; ++i) {
            const unsigned NEXT_LAG = time_lag + dl1.LAGS[i];
            if (NEXT_LAG >= SEQUENCE_LENGTH)  // ignore, if index exceeds sequence length (i.e. ignore y(t) if t<0)
                break;
            gradients[NEXT_LAG] += temporary_gradient * dl1.AR_kernels[i];
        }

    }

}


void TRRL::extract_contribution(FirstLayer& layer,
                                const std::vector<VectorClass>& layer_input,
                                const std::vector<VectorClass>& recurrent_history,
                                const VectorClass& gradient_gi,
                                const unsigned TIME_LAG) {

    // kernel
    for (unsigned j = 0; j < layer.OUTPUT_SZ; ++j) {
        // base index (j rows have already been processed)
        const unsigned BASE = j * layer.INPUT_SZ;
        const unsigned ARRAY_INDEX = SEQUENCE_LENGTH - TIME_LAG - 1;  // index in the history array
        for (unsigned k = 0; k < layer.INPUT_SZ; ++k)
            layer.gradient_params(BASE + k) += gradient_gi(j) * layer_input[ARRAY_INDEX].at(k);
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
                    layer.gradient_params(BASE + BASE_KERNELS + k) +=
                            gradient_gi(j) * recurrent_history[ARRAY_INDEX].at(k);
            }
        }
    }

    // bias
    // base index for bias vector (all kernels have already been processed)
    const unsigned BASE_BIAS = (layer.INPUT_SZ + layer.N_LAGS * NN::OUTPUT_NODES) * layer.OUTPUT_SZ;
    for (unsigned j = 0; j < layer.OUTPUT_SZ; ++j) {
        layer.gradient_params(BASE_BIAS + j) += gradient_gi(j);
    }

}
