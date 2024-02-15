/// \a Pure \a abstract \a class that serves as a blueprint for any \a optimizer.

#ifndef ABSTRACTSOLVER_H
#define ABSTRACTSOLVER_H

#include "../global.h"

#include <fstream>

#include "../Layers/FirstLayer.h"
#include "../Layers/SecondLayer.h"

class AbstractSolver {

    /********************************************** ATTRIBUTES *********************************************/

public:

    /// Name of the solver (e.g. Adam, SGD, ...)
    /**
     *  It is set during the construction of the object and cannot be modified later.
     */
    const std::string SOLVER_NAME;


    /// Decay parameter for learning rate of the solver.
    /**
     * Concerns the reduction of the learning rate while epochs increase. The following schedule is applied:
     *  \f[
     *      \mu_t = \frac{ \mu_0 } {1 + t * DECAY }
     *  \f]
     *  where t represents the current epoch.
     */
    const NN::type LR_DECAY;

    /// Initial learning rate of the solver.
    /**
     * Concerns the step-size of the correction induced by the gradient in the stochastic descent.
     */
    const NN::type INITIAL_LEARNING_RATE;


    /// Current learning rate of the solver.
    /**
     * According to decay schedule.
     */
    NN::type current_LR;

    /// Vector of update of the first layer parameters, to be applied when required by the optimizer.
    /**
     * The updates are stored while processing the sequences.
     */
    VectorClass dl1_weights_update;

    /// Vector of update of the second layer parameters, to be applied when required by the optimizer.
    /**
     * The updates are stored while processing the sequences.
     */
    VectorClass dl2_weights_update;

    /********************************************** CONSTRUCTORS **********************************************/

public:

    /// Constructor of the object.
    /**
     * @param dl1_NTP:                 Number of Trainable parameters of First Layer
     * @param dl2_NTP:                 Number of Trainable parameters of Second Layer
     * @param INITIAL_LEARNING_RATE_:  Learning Rate for the solver
     * @param decay_:                  Decay Rate for learning rate
     */
    AbstractSolver(std::string SOLVER_NAME_,
                   unsigned dl1_NTP, unsigned dl2_NTP,
                   NN::type INITIAL_LEARNING_RATE_, NN::type DECAY_ = 0);


    /***************************************** METHODS ************************************************/

public:

    /// Adjust Learning Rate (according to decay schedule)
    /**
     * @param current_epoch:    current epoch of training
     */
    void adjust_LR(unsigned current_epoch);


    /// Compute the adjustment of weights required by the selected solver (abstract method)
    /**
     * Note: here, we compute and store only the updates that must be applied.
     * In other words, we store here dW such that the new weights are W(new) = W(old) - dW.
     * @param dl1_gt: vector representing the updates of the first vector of weights
     * @param dl2_gt: vector representing the updates of the second vector of weights
     */
    virtual void solve(const VectorClass& dl1_gt, const VectorClass& dl2_gt) = 0;

};


#endif //ABSTRACTSOLVER_H
