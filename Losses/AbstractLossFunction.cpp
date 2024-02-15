
#include "AbstractLossFunction.h"

/********************************************** CONSTRUCTORS ************************************************/

AbstractLossFunction::AbstractLossFunction(unsigned INPUT_SZ_, std::string LOSS_NAME_) :
        INPUT_SZ(INPUT_SZ_), output(0), LOSS_NAME(LOSS_NAME_),
        gradient(VectorClass(INPUT_SZ_, 0))  //zero initialization
        {}
