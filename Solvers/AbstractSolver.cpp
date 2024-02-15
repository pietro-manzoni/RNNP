#include "AbstractSolver.h"

#include <utility>

AbstractSolver::AbstractSolver(std::string SOLVER_NAME_, unsigned dl1_NTP, unsigned dl2_NTP,
          NN::type INITIAL_LEARNING_RATE_, NN::type LR_DECAY_) :

          SOLVER_NAME(std::move(SOLVER_NAME_)),

          INITIAL_LEARNING_RATE(INITIAL_LEARNING_RATE_),
          current_LR(INITIAL_LEARNING_RATE_),
          LR_DECAY(LR_DECAY_),

          dl1_weights_update(dl1_NTP),
          dl2_weights_update(dl2_NTP) {}


void AbstractSolver::adjust_LR(unsigned current_epoch){
    current_LR = INITIAL_LEARNING_RATE /
        (1 + static_cast<NN::type>(current_epoch) * LR_DECAY);
}
