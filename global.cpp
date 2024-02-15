#include "global.h"
#include <limits>

// NN output features (default 1)
unsigned NN::OUTPUT_NODES = 2;

// print format (defaults)
int NN::TRAINING_LOG_DIGITS = 3;
int NN::PREDICTION_DIGITS = 10;
int NN::SPACING = 11;

// utilities
const NN::type NN::infty = std::numeric_limits<NN::type>::infinity();
const NN::type NN::NaN = std::numeric_limits<NN::type>::quiet_NaN();
