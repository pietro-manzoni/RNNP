#ifndef DNAX_LIBRARY_RNNMODEL_H
#define DNAX_LIBRARY_RNNMODEL_H

#include <memory>
#include <vector>
#include <numeric>
#include <random>

#include "../global.h"

#include "../ActivationFunctions/AbstractActivationFunction.h"
#include "../ActivationFunctions/ExpLU.h"
#include "../ActivationFunctions/Linear.h"
#include "../ActivationFunctions/LeakyReLU.h"
#include "../ActivationFunctions/ReLU.h"
#include "../ActivationFunctions/Sigmoid.h"
#include "../ActivationFunctions/Softmax.h"
#include "../ActivationFunctions/Softsign.h"
#include "../ActivationFunctions/Swish.h"
#include "../ActivationFunctions/Tanh.h"

#include "../DataStructures/VectorClass.h"
#include "../DataStructures/Matrix.h"

#include "../GradientAlgorithms/AbstractGradientAlgo.h"
#include "../GradientAlgorithms/BPTT.h"
#include "../GradientAlgorithms/RTRL.h"
#include "../GradientAlgorithms/TRRL.h"

#include "../Layers/FirstLayer.h"
#include "../Layers/SecondLayer.h"

#include "../Losses/AbstractLossFunction.h"
#include "../Losses/LogLikelihood.h"
#include "../Losses/GaussianCRPS.h"
#include "../Losses/GaussianCRPSLambda.h"
#include "../Losses/GaussianCRPSTrivial.h"
#include "../Losses/GaussianWinkler.h"
#include "../Losses/GaussianWinklerTrivial.h"
#include "../Losses/MSE.h"

#include "../Metrics/AbstractMetric.h"
#include "../Metrics/MSE_metric.h"

#include "../Solvers/AbstractSolver.h"
#include "../Solvers/Adam.h"
#include "../Solvers/Nadam.h"

#include "../utilities.h"

#include <mpi.h>



class RnnModel{

    /********************************************** ATTRIBUTES **************************************************/

protected:

    FirstLayer dl1;
    SecondLayer dl2;

    std::shared_ptr<AbstractActivationFunction> af1_ptr = nullptr;
    std::shared_ptr<AbstractLossFunction> loss_ptr = nullptr;
    std::shared_ptr<AbstractMetric> metric_ptr = nullptr;
    std::shared_ptr<AbstractGradientAlgo> gradient_algorithm_ptr = nullptr;
    std::shared_ptr<AbstractSolver> optimizer_ptr = nullptr;

    unsigned PATIENCE;
    NN::type MIN_DELTA;
    bool RESTORE_BEST;

    unsigned NUM_EPOCHS;
    unsigned BATCH_SIZE;

    int rank;
    int size;

    /****************************************** CONSTRUCTORS ********************************************/

public:

    /// Constructor of the object
    RnnModel(unsigned INPUT_NEURONS, unsigned HIDDEN_NEURONS,
             unsigned OUTPUT_NEURONS, const std::vector<unsigned>& WHICH_LAGS,
             unsigned DL1_SEED = 0, unsigned DL2_SEED = 0);


    /********************************************** METHODS *****************************************************/

protected:

  /// Set the default options for the training, if they are missing (not specified)
  void check_pointers() const;

public:

    /// Set the optimizer for the training
    /**
     * @param ACTIVATION_FUNCTION:  Name of the Activation function.
     */
    void set_activation(const std::string& ACTIVATION_FUNCTION);

    /// Set the loss function for the training
    /**
     * @param LOSS:           Name of the Loss function.
     * @param LAMBDA:         Tail decay (if applies)
     */
    void set_loss(const std::string& LOSS, NN::type LAMBDA = 1.0);

    /// Set the metric for the training
    /**
     * @param METRIC:  Name of the Metric. "None" is accepted.
     */
    void  set_metric(const std::string& METRIC);

    /// Set the metric for the training
    /**
     * @param ALGORITHM: Name of the gradient algorithm.
     *                   Can be "TRRL", "RTRL" or "BPTT".
     */
     void set_gradientalgo(const std::string& ALGORITHM, unsigned SEQUENCE_LENGTH);

    /// Set the optimizer for the training
    /**
     * @param SOLVER:           Name of the Solver. Can be "Adam", "Nadam"
     * @param LEARNING_RATE:    Learning Rate used during the training
     * @param NUM_EPOCHS:       Maximum number of epochs
     * @param BATCH_SIZE:       Size of each Mini-Batch
     * @param LR_DECAY:         Learning Rate Decay Coefficient
     */
    void set_optimizer(const std::string& SOLVER, NN::type LEARNING_RATE,
        unsigned NUM_EPOCHS, unsigned BATCH_SIZE, NN::type LR_DECAY = 0.);

    /// Set the optimizer for the training
    /**
     * @param PATIENCE_:     Patience parameters (epochs without improvements)
     * @param MIN_DELTA_:    Minimum variation of improvement
     * @param RESTORE_BEST:  Restore best status (i.e. the minimum for the metric)
     */
    void add_stopping_criterion(unsigned PATIENCE_, NN::type MIN_DELTA_, bool RESTORE_BEST_ = true);

    /// Set paramters for MPI parallelization
    /**
     * @param rank_:     rank of the process
     * @param size_:     size of common world
     */
    void mpi_parallelize(int rank_, int size_);

    /// Create Checkpoint
    /**
     * Save current weights of the dense layers.
     */
    void create_checkpoint();

    // Train RNN Model
    void fit(const std::vector< std::vector<VectorClass> >& x_sequences_train,
        const std::vector<NN::type>& y_sequences_train,
        std::ofstream& output_log, unsigned SHUFFLING_SEED = 0);

    std::vector<VectorClass> predict(const std::vector< std::vector<VectorClass> >& x_sequences);

    std::pair<NN::type, NN::type> evaluate_accuracy(const std::vector< std::vector<VectorClass> >& x_sequences,
        const std::vector<NN::type>& y_sequences);
};


#endif //DNAX_LIBRARY_RNNMODEL_H
