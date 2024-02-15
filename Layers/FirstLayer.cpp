#include "FirstLayer.h"

#include <algorithm>



/********************************************** CONSTRUCTORS ************************************************/

FirstLayer::FirstLayer(unsigned INPUT_SZ_, unsigned OUTPUT_SZ_,
                        const std::vector<unsigned>& LAGS_, unsigned random_seed) :

        INPUT_SZ(INPUT_SZ_),
        OUTPUT_SZ(OUTPUT_SZ_),
        N_LAGS(static_cast<unsigned>(LAGS_.size())),
        LAGS(LAGS_),
        MAX_LAG( LAGS_.empty() ? 0 : *std::max_element(LAGS_.begin(),LAGS_.end()) ),
        NTP( OUTPUT_SZ_ * (INPUT_SZ_ + NN::OUTPUT_NODES * N_LAGS + 1) ),

        // initialize weights
        kernel(Matrix( OUTPUT_SZ_, INPUT_SZ_, 'n', random_seed++, 0.0, 0.1 )),
        bias(VectorClass( OUTPUT_SZ_, 0)),  // null initialization
        output(VectorClass(OUTPUT_SZ_, 0)),
        gradient_params( VectorClass(NTP, 0) ),

        // initialize best weights
	      kernel_best(kernel),
	      bias_best(bias)
    {

        //initializing the vector of Autoregressive Kernels
        for (unsigned i = 0; i < N_LAGS; ++i) {
            // set an initial condition for the AR kernels
            AR_kernels.emplace_back(      Matrix(OUTPUT_SZ_, NN::OUTPUT_NODES, 'u', random_seed  , 0.0, 0.0 ) );
            AR_kernels_best.emplace_back( Matrix(OUTPUT_SZ_, NN::OUTPUT_NODES, 'u', random_seed++, 0.0, 0.0 ) );
        }

    }


/********************************************** METHODS *****************************************************/


void FirstLayer::update_output(const VectorClass& input,
                                const std::vector<VectorClass>& output_history, const unsigned T){

    // computing contribution of kernel
    mult_in_place(kernel, input, output);

    // computing contribution of autoregressive terms
    for (unsigned i = 0; i < N_LAGS; ++i)
        //if(IDX-LAGS[i] < output_history.size()) // avoid undefined behaviour (unsigned)
        if( T >= LAGS[i]) // if index is admissible
          output += AR_kernels[i] * output_history[ T - LAGS[i] ];
    // adding bias
    output += bias;

}


void FirstLayer::update_output(const VectorClass& input,
                               const CircularBuffer<VectorClass>& output_history){

    // computing contribution of kernel
    mult_in_place(kernel, input, output);

    // computing contribution of autoregressive terms
    for (unsigned i = 0; i < N_LAGS; ++i)
        output += AR_kernels[i] * output_history( LAGS[i] );

    // adding bias
    output += bias;

}


void FirstLayer::new_weights(const VectorClass &new_wgt){

    // update the weights of the kernel
    for (unsigned i = 0; i < OUTPUT_SZ; ++i) {
        const unsigned BASE = i * INPUT_SZ;
        for (unsigned j = 0; j < INPUT_SZ; ++j) {
            bool sgn = kernel(i, j) > 0;
            kernel(i, j) *= (1 - LAMBDA2);                  // L2 regularization
            kernel(i, j) -= sgn ? (LAMBDA1) : (-LAMBDA1);   // L1 regularization
            kernel(i, j) -= new_wgt(BASE + j);
        }
    }

    // update the weights of the AR_kernels
    for (unsigned l = 0; l < N_LAGS; ++l) {
        const unsigned BASE2 = (INPUT_SZ + l * NN::OUTPUT_NODES) * OUTPUT_SZ;
        for (unsigned i = 0; i < OUTPUT_SZ; ++i) {
            const unsigned BASE = i * NN::OUTPUT_NODES;
            for (unsigned j = 0; j < NN::OUTPUT_NODES; ++j) {
                bool sgn = AR_kernels[l](i, j) > 0;
                AR_kernels[l](i, j) *= (1 - LAMBDA2);                   // L2 regularization
                AR_kernels[l](i, j) -= sgn ? (LAMBDA1) : (-LAMBDA1);   // L1 regularization
                AR_kernels[l](i, j) -= new_wgt(BASE2 + BASE + j);
            }
        }
    }

    // update the weights of the bias
    const unsigned BASE3 = (INPUT_SZ + N_LAGS * NN::OUTPUT_NODES) * OUTPUT_SZ;
    for (unsigned i=0; i < OUTPUT_SZ; ++i) {
        //bool sgn = bias(i) > 0;  // NB: usually regularization is not applied to bias (cf. Goodfellow p.230)
        //bias(i) *= (1 - learning_rate * LAMBDA2);                                     // L2 regularization
        //bias(i) -= sgn ? (learning_rate * LAMBDA1) : (- learning_rate * LAMBDA1);     // L1 regularization
        bias(i) -= new_wgt(BASE3 + i);
    }

}

void FirstLayer::add_regularization(NN::type L1_regularization, NN::type L2_regularization){

    // modifying members
    LAMBDA1 = L1_regularization;
    LAMBDA2 = L2_regularization;

}


void FirstLayer::print() const{

    std::cout << "--------- Layer Weights ---------\n" << std::endl;

    // print kernel
    kernel.print();

    // print the autoregressive kernels
    for (unsigned i = 0; i < N_LAGS; ++i)
        AR_kernels[i].print();

    // print bias
    bias.print();

}

void FirstLayer::save_status(){

    // store best configurations
    kernel_best = kernel;
    bias_best = bias;
    for (unsigned i = 0; i < N_LAGS; ++i)
        AR_kernels_best[i] = AR_kernels[i];

}


void FirstLayer::restore_status(){

    kernel = kernel_best;
    bias = bias_best;
    for (unsigned i = 0; i < N_LAGS; ++i)
        AR_kernels[i] = AR_kernels_best[i];

}
