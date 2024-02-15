#include "AbstractGradientAlgo.h"

AbstractGradientAlgo::AbstractGradientAlgo(std::string ALGO_NAME_,
    FirstLayer& dl1_, SecondLayer& dl2_,
    unsigned SEQUENCE_LENGTH_) :
    ALGO_NAME(std::move(ALGO_NAME_)),
    dl1(dl1_), dl2(dl2_),
    SEQUENCE_LENGTH(SEQUENCE_LENGTH_),

    //instantiate circular buffers for storing buffer
    output_buffer(dl1.MAX_LAG)

{
    for (unsigned i = 0; i < dl1.MAX_LAG; ++i)
        output_buffer.insert(dl2.output);
}


void AbstractGradientAlgo::predict(const std::vector<VectorClass>& x_sequence_train) {

    // reset history
    output_buffer.reset();

    for (unsigned t = 0; t < SEQUENCE_LENGTH; ++t) {

        // forward flow through the network
        dl1.update_output(x_sequence_train[t], output_buffer);
        af1->update_output(dl1.output);
        dl2.update_output(af1->output, output_buffer);

        //update stored values and Jacobian matrices
        output_buffer.insert(dl2.output);
    }

}


void AbstractGradientAlgo::set_dependencies(std::shared_ptr<AbstractActivationFunction> af1_ptr_,
    std::shared_ptr<AbstractLossFunction> loss_ptr_){

      af1 = af1_ptr_;
      loss = loss_ptr_;

}
