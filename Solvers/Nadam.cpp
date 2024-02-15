
#include <cmath>
#include "Nadam.h"


/********************************************** CONSTRUCTORS ************************************************/

Nadam::Nadam(unsigned dl1_NTP, unsigned dl2_NTP, NN::type learning_rate_, NN::type decay_) :
    AbstractSolver("Nadam", dl1_NTP, dl2_NTP, learning_rate_, decay_),

    dl1_mean(dl1_NTP), dl1_variance(dl1_NTP),
    dl2_mean(dl2_NTP), dl2_variance(dl2_NTP) {}


/********************************************** METHODS *****************************************************/

void Nadam::solve(const VectorClass& dl1_gt, const VectorClass& dl2_gt) {

    // Update variables according to Nadam/Adam definition
    ++t;
    NN::type alpha_t = current_LR * sqrt(1 - pow(beta_2, t)) / (1-pow(beta_1, t));

    // ------------------------------------  First Layer ---------------------------------------- //

    // Adjust the rolling mean
    dl1_mean *= beta_1;
    dl1_mean += one_minus_beta_1 * dl1_gt;

    // Adjust the rolling variance
    dl1_variance *= beta_2;
    dl1_variance += one_minus_beta_2 * pow2(dl1_gt);

    // Update kernels and bias
    dl1_weights_update = alpha_t * ((beta_1*dl1_mean) - (one_minus_beta_1*dl1_gt)) / ( sqrt(dl1_variance) + epsilonhat );

    // ------------------------------------  Second Layer ---------------------------------------- //

    // Adjust the rolling mean
    dl2_mean *= beta_1;
    dl2_mean += one_minus_beta_1 * dl2_gt;

    // Adjust the rolling variance
    dl2_variance *= beta_2;
    dl2_variance += one_minus_beta_2 * pow2(dl2_gt);

    // Update kernels and bias
    dl2_weights_update = alpha_t * ((beta_1*dl2_mean) - (one_minus_beta_1*dl2_gt)) / ( sqrt(dl2_variance) + epsilonhat );

}
