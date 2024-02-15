/// %Nadam solver
/**
 * Class that implements the optimization operations that are performed during the training
 * phase of a Neural Network. For further explanation of the algorithms, cf. Kingma-Ba (2014)
 */

#ifndef Nadam_H
#define Nadam_H


#include "AbstractSolver.h"

#include "../DataStructures/Matrix.h"
#include "../DataStructures/VectorClass.h"
#include "../DataStructures/AlgebraicOperations.h"

#include "../Layers/FirstLayer.h"


class Nadam : public AbstractSolver{

    /********************************************** ATTRIBUTES **************************************************/

private:

    /// Exponential decay rate for the 1st moment estimates.
    const NN::type beta_1 = 0.9;

    /// Exponential decay rate for the 2nd moment estimates.
    const NN::type beta_2 = 0.999;

    /// Small constant for numerical stability.
    /**
     * This epsilon is "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1),
     * not the epsilon in Algorithm 1 of the paper.
     */
    const NN::type epsilonhat = 1e-10;

    /// Time bias-correction
    /**
     * Needed for Adam-like mean correction. Must be increased after each mini-batch has been processed.
     */
    unsigned t = 0;

    /// Sequential counter
    /**
     * Counts how many elements are contained in a batch.
     */
    unsigned counter = 0;

    /// Useful variable
    /**
     * Stored for convenience.
     */
    const NN::type one_minus_beta_1 = 1 - beta_1;

    /// Useful variable
    /**
     * Stored for convenience.
     */
    const NN::type one_minus_beta_2 = 1 - beta_2;

    /// Rolling mean of First FirstLayer
    VectorClass dl1_mean;

    /// Rolling mean of Second FirstLayer
    VectorClass dl2_mean;

    /// Rolling variance of First FirstLayer
    VectorClass dl1_variance;

    /// Rolling variance of Second FirstLayer
    VectorClass dl2_variance;


    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Constructor of the object.
    /**
     * Read all the elements contained in a given .csv file and store them in the #data structure.
     * @param dl1_NTP:          Number of Trainable parameters of First Layer
     * @param dl2_NTP:          Number of Trainable parameters of Second Layer
     * @param learning_rate_:   Learning Rate for the solver
     * @param decay_:           Decay Rate for learning rate
     */
    Nadam(unsigned dl1_NTP, unsigned dl2_NTP, NN::type learning_rate_, NN::type decay_ = 0);


    /********************************************** METHODS *****************************************************/

public:

    /// Apply Nadam solver and update the status of the object.
    /**
     * Overriding from abstract class
     */
    void solve(const VectorClass& dl1_gt, const VectorClass& dl2_gt) override;

};


#endif //Nadam_H
