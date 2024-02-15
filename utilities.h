#ifndef UTILITIES_H
#define UTILITIES_H

#include "DataStructures/CircularBuffer.h"
#include "DataStructures/Matrix.h"
#include "DataStructures/VectorClass.h"
#include "DataStructures/AlgebraicOperations.h"

#include "Layers/FirstLayer.h"

#include "ActivationFunctions/AbstractActivationFunction.h"
#include "Metrics/AbstractMetric.h"
#include "Losses/AbstractLossFunction.h"
#include "Solvers/AbstractSolver.h"


/// Split time series into suitable subsequences to be fed to RNN
/**
 *
 * @param x_dataset:        dataset containing the entire sequence of exogenous inputs
 * @param y_dataset:        dataset containing the entire sequence of target values
 * @param x_sequences:      container where the exogenous sequences are stored
 * @param y_sequences:      container where the targets
 * @param WINDOW_SIZE:      length of each subsequence
 * @param TIME_DISTANCE:    distance in time between two consecutive subsequences
 */
void split_dataset(const std::vector<VectorClass>& x_dataset,
                   const std::vector<NN::type>& y_dataset,
                   std::vector< std::vector<VectorClass> >& x_sequences,
                   std::vector<NN::type>& y_sequences,
                   const unsigned WINDOW_SIZE, const unsigned TIME_DISTANCE);


/// Compute Quantile of a LOGNORMAL distribution
/**
 */
NN::type compute_quantile(NN::type mu, NN::type sigma, NN::type alpha);

/// Compute Two-sided Confidence Interval of a LOGNORMAL distribution
/**
 */
std::pair<NN::type, NN::type> compute_ci(NN::type mu, NN::type sigma, NN::type alpha);

/// Compute Statistics
/**
 */
NN::type compute_statistics(const std::vector<NN::type>& mu, const std::vector<NN::type>& sigma,
    const std::vector<NN::type>& point_forecast, const std::vector<NN::type>& consumption_true,
    std::ofstream& output_stats);


NN::type normpdf(NN::type x);

NN::type normcdf(NN::type x);

NN::type norminv(NN::type alpha, NN::type tol = 1e-10, unsigned max_iter = 1e3);

NN::type integratelogQ(NN::type t);


#endif //UTILITIES_H
