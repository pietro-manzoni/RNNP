// Global variables used throughout the code

#include <iostream>

/// Collect the global variables used throughout the code
namespace NN{

    // NN features
    /// Type used in the code
    typedef double type;
    /// Size of output vector, the one which are used in the auto-regression (2, in the usual case)
    extern unsigned OUTPUT_NODES;

    // print format
    /// Number of decimal digits to be shown for training (loss & metrics)
    extern int TRAINING_LOG_DIGITS;
    /// Number of decimal digits to be shown for fitted/predicted values
    extern int PREDICTION_DIGITS;
    /// Spacing in the 'std::cout' printing
    extern int SPACING;

    // utilities
    /// Infinity, for brevity
    extern const type infty;
    /// Nan, for brevity
    extern const type NaN;

}
