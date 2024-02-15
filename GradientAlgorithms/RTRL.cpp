#include "RTRL.h"

RTRL::RTRL(FirstLayer& dl1_, SecondLayer& dl2_,
           unsigned SEQUENCE_LENGTH_) :
           AbstractGradientAlgo("RTRL", dl1_, dl2_, SEQUENCE_LENGTH_),

           // zero initialize the Jacobian matrices
           dl1_jacobian_params(Matrix(dl2.OUTPUT_SZ, dl1.NTP, 0)),
           dl2_jacobian_params(Matrix(dl2.OUTPUT_SZ, dl2.NTP, 0)),

           //instantiate circular buffers for storing buffer
           dtheta_buffer(dl1.MAX_LAG),
           dphi_buffer(dl1.MAX_LAG)

{
    // and initialize them with Vector/Matrices of zeros of the right size
    for (unsigned i = 0; i < dl1.MAX_LAG; ++i) {
        dtheta_buffer.insert(dl1_jacobian_params);
        dphi_buffer.insert(dl2_jacobian_params);
    }
}



void RTRL::process_sequence(const std::vector<VectorClass>& x_sequence_train, const NN::type y_real){

    // reset buffer
    output_buffer.reset();
    dtheta_buffer.reset();
    dphi_buffer.reset();

    for (unsigned t = 0; t < SEQUENCE_LENGTH; ++t) {

        // forward flow through the network
        dl1.update_output(x_sequence_train[t], output_buffer);
        af1->update_output(dl1.output);
        dl2.update_output(af1->output, output_buffer);

        // compute Jacobian matrix of activation function
        af1->update_jacobian(dl1.output);

        // initialize matrix
        Matrix K2A1(dl2.kernel.rows(), dl2.kernel.cols());

        // obtain matrix used as factor
        if (af1->DIAGONAL)
            mult_AD(dl2.kernel, af1->jacobian, K2A1);
        else
            mult_in_place(dl2.kernel, af1->jacobian, K2A1);

        // partial derivative contributions
        dl1_partial_derivative(x_sequence_train[t], t, K2A1);
        dl2_partial_derivative();

        for (unsigned i = 0; i < dl1.N_LAGS; ++i){
            // store for convenience
            Matrix tmp_mtrx = K2A1 * dl1.AR_kernels[i];

            //dl1_jacobian_params += tmp_mtrx * dtheta_buffer(dl1.LAGS[i]);
            mult_add_in_place(tmp_mtrx, dtheta_buffer(dl1.LAGS[i]), dl1_jacobian_params);

            //dl2_jacobian_params += tmp_mtrx * dphi_buffer(dl1.LAGS[i]);
            mult_add_in_place(tmp_mtrx, dphi_buffer(dl1.LAGS[i]), dl2_jacobian_params);
        }

        //update stored values and Jacobian matrices
        output_buffer.insert(dl2.output);
        dtheta_buffer.insert(dl1_jacobian_params);
        dphi_buffer.insert(dl2_jacobian_params);

    }

    // compute gradient of loss with respect to last output
    loss->update_gradient(dl2.output, y_real);

    // compute overall gradient
    mult_in_place(loss->gradient, dl1_jacobian_params, dl1.gradient_params);
    mult_in_place(loss->gradient, dl2_jacobian_params, dl2.gradient_params);

}

// Partial Derivative for First Layer
void RTRL::dl1_partial_derivative(const VectorClass& x_sequence, const unsigned T, const Matrix& mtrx){

    // reset jacobian params
    dl1_jacobian_params.reset();

    // kernel
    for (unsigned i = 0; i < NN::OUTPUT_NODES; ++i){
        for (unsigned j = 0;  j < dl1.OUTPUT_SZ; ++j){
            // base index (j rows have already been processed)
            const unsigned BASE = j * dl1.INPUT_SZ;
            const NN::type CACHE_VAL = mtrx(i,j);
            for (unsigned k = 0; k < dl1.INPUT_SZ; ++k){
                dl1_jacobian_params(i, BASE + k) = CACHE_VAL * x_sequence(k);
            }
        }
    }

    // AR kernels
    for (unsigned l = 0; l < dl1.N_LAGS; ++l) {
        // base index (some kernels have already been processed)
        const unsigned BASE_KERNELS = (dl1.INPUT_SZ + l * NN::OUTPUT_NODES) * dl1.OUTPUT_SZ;
        for (unsigned i = 0; i < NN::OUTPUT_NODES; ++i){
            for (unsigned j = 0; j < dl1.OUTPUT_SZ; ++j) {
                // base index (for this kernel, j rows have already been processed)
                const unsigned BASE = j * NN::OUTPUT_NODES;
                const NN::type CACHE_VAL = mtrx(i, j);
                for (unsigned k = 0; k < NN::OUTPUT_NODES; ++k){
                    dl1_jacobian_params(i, BASE + BASE_KERNELS + k) = CACHE_VAL * output_buffer(dl1.LAGS[l]).at(k);
                }
            }
        }
    }

    // bias
    // base index for bias vector (all kernels have already been processed)
    const unsigned BASE_BIAS = (dl1.INPUT_SZ + dl1.N_LAGS * NN::OUTPUT_NODES) * dl1.OUTPUT_SZ;
    for (unsigned i = 0; i < NN::OUTPUT_NODES; ++i) {
        for (unsigned j = 0;  j < dl1.OUTPUT_SZ; ++j) {
            dl1_jacobian_params(i, BASE_BIAS + j) = mtrx(i,j);
        }
    }

}


// Partial Derivative for Second Layer
void RTRL::dl2_partial_derivative(void){

    // reset jacobian params
    dl2_jacobian_params.reset();

    // kernel
    for (unsigned i = 0; i < dl2.OUTPUT_SZ; ++i){
        const unsigned BASE = i * dl2.INPUT_SZ;
        for (unsigned j = 0; j < dl2.INPUT_SZ; ++j){
            dl2_jacobian_params(i, BASE+j) = af1->output(j);
        }
    }

    // bias
    const unsigned BASE_BIAS = (dl2.INPUT_SZ + dl2.N_LAGS * NN::OUTPUT_NODES) * dl2.OUTPUT_SZ;
    for (unsigned i = 0; i < dl2.OUTPUT_SZ; ++i)
        dl2_jacobian_params(i, BASE_BIAS + i) = 1;

}
