/// Dense Layer that admits any recurrent connection
/**
 * Class that implements a Dense Layer (the affine map only), which is composed of
 * a certain number of neurons (the state of the layer). This class deals with
 * storing and updating all the necessary weights associated with every
 * regular or feedback connection.
 */


#ifndef FirstLayer_H
#define FirstLayer_H

#include "../global.h"

#include <fstream>

#include "../DataStructures/CircularBuffer.h"
#include "../DataStructures/Matrix.h"
#include "../DataStructures/VectorClass.h"
#include "../DataStructures/AlgebraicOperations.h"


class FirstLayer {

    /***************************************** ATTRIBUTES *********************************************/

private:

    friend class AbstractSolver;

public:

    /// Input size of the layer
    /**
     * It considers all the \a exogenous (i.e. non-recurrent) inputs that are provided
     * to the layer.
     * It is set during the construction of the object and cannot be modified later.
     */
    const unsigned INPUT_SZ;

    /// Output size of the layer
    /**
     * Actually, it represents the number of neurons that compose the layer.
     * It is set during the construction of the object and cannot be modified later.
     */
    const unsigned OUTPUT_SZ;

    /// Number of recurrent inputs
    /**
     * It counts how many autoregressive Jordan's inputs are considered. If for instance the
     * layer is fed with the 1-lag, the 3-lag and the 4-lag output, N_LAGS is equal
     * to 3 = #{1,3,4}. Thus, it only concerns the cardinality.
     */
    const unsigned N_LAGS;

    /// Which lags have to be considered?
    /**
     * std::vector that contains which lags have to be included as Jordan's recurrent inputs.
     * #N_LAGS is equal to the cardinality of this vector.
     */
    const std::vector<unsigned> LAGS; //which lags have to be included

    /// Which is the maximum lag that we have to remember?
    /**
     * Maximum lag that has to be considered. This affects the size of each
     * CircularBuffer, since it is useless to remember too many lags.
     */
    const unsigned MAX_LAG; //which is the maximum lag?

    /// Number of trainable parameters
    /**
     * Total number of weights of the layer. This accounts for the "regular" kernel
     * and bias, plus all the autoregressive kernels.
     */
    const unsigned NTP; //number of trainable parameters

    /// Coefficient for L1 weights regularization
    /**
     * Regularization coefficient. A penalty is added to the loss function: this is
     * equal to LAMBDA1 times the L1 norm of the weights.
     * (cf. Goodfellow, p.234)
     */
    NN::type LAMBDA1 = 0; //L1 weights regularization

    /// Coefficient for L2 weights regularization
    /**
     * Regularization coefficient. A penalty is added to the loss function: this is
     * equal to LAMBDA2 times one half of the L2 squared norm of the weights.
     * (cf. Goodfellow, p.231)
     */
    NN::type LAMBDA2 = 0; //L2 weights regularization

    /// Kernel of the Dense Layer.
    /**
     * It concerns the linear application from the \a exogenous inputs to
     * the neurons of the layer. No bias or recurrent connections are here considered.
     */
    Matrix kernel;

    /// Autoregressive kernels of the Dense Layer.
    /**
     * It concerns all the linear applications from each \a recurrent input to
     * the neurons of the layer. Each recurrent input has its own autoregressive kernel.
     * Thus there are as many autoregressive kernels as N_LAGS.
     */
    std::vector<Matrix> AR_kernels;

    /// Bias of the layer.
    /**
     * It concerns the contribution of the bias in the application from all the inputs to the
     * neurons. Each neuron of the layer has its own bias: thus it is a vector of size #OUTPUT_SZ.
     */
    VectorClass bias;

    /// Best Kernel of the Dense Layer.
    /**
     * Stored for restoring best condition after Early Stopping.
     */
    Matrix kernel_best;

    /// Best AR_Kernels of the Dense Layer.
    /**
     * Stored for restoring best condition after Early Stopping.
     */
    std::vector<Matrix> AR_kernels_best;

    /// Best Bias of the Dense Layer.
    /**
     * Stored for restoring best condition after Early Stopping.
     */
    VectorClass bias_best;

    /// Output of the layer (before applying activation function).
    /**
     * It represents the state of the neurons before applying the
     * activation function.
     */
    VectorClass output;


    /// Gradient of the loss with respect to the trainable parameters.
    /**
     * Gradient used in the optimization procedure.
     */
    VectorClass gradient_params;

    /***************************************** CONSTRUCTORS *******************************************/

public:

    /// Constructor of the object
    /**
     * @param INPUT_SZ_:            number of pure \a exogenous inputs
     * @param OUTPUT_SZ_:           number of neurons of the layer
     * @param LAGS_:                vector of recurrent lags (also an initializer list is accepted)
     * @param random_seed:          random seed
     */
    FirstLayer(unsigned INPUT_SZ_, unsigned OUTPUT_SZ_,
                const std::vector<unsigned>& LAGS_, unsigned random_seed);

    /****************************************** METHODS *************************************************/

public:

    /// Perform forward pass through neurons (BPTT, TRRL)
    /**
     * @param input:                exogenous inputs
     * @param output_history:       vector containing the entire sequence of previous outputs
     * @param T:                    current time (processed units)
     */
    void update_output(const VectorClass& input,
      const std::vector<VectorClass>& output_history,
      const unsigned T);

    /// Perform forward pass through neurons  (RTRL)
    /**
     * @param input:                exogenous inputs
     * @param output_history:       circular buffer containing the previous outputs (just the useful ones)
     */
    void update_output(const VectorClass& input,
                                   const CircularBuffer<VectorClass>& output_history);

    /// Update weights
    /**
     * @param learning_rate:    learning rate of solver
     * @param new_wgt:          correction of the current condition according to the solver algorithm
     */
    void new_weights(const VectorClass& new_wgt);


    /// Add Regularization
    /**
     * @param L1_regularization:   L1-regularization coefficient
     * @param L2_regularization:   L2-regularization coefficient
     */
    void add_regularization(NN::type L1_regularization, NN::type L2_regularization);


    /// Print state of the layer
    void print(void) const;

    /// Save current status of the layer weights.
    /**
     * Used for creating checkpoints.
     */
    void save_status(void);

    /// Restore optimal status in case of Early exit.
    void restore_status(void);

};


#endif //FirstLayer_H
