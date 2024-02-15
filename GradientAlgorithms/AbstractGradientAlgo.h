#ifndef RNN_P__LIBRARY_ABSTRACTGRADIENTALGO_H
#define RNN_P__LIBRARY_ABSTRACTGRADIENTALGO_H

#include <memory>

#include "../DataStructures/CircularBuffer.h"
#include "../DataStructures/Matrix.h"
#include "../DataStructures/VectorClass.h"
#include "../DataStructures/AlgebraicOperations.h"

#include "../Layers/FirstLayer.h"
#include "../Layers/SecondLayer.h"

#include "../ActivationFunctions/AbstractActivationFunction.h"
#include "../Metrics/AbstractMetric.h"
#include "../Losses/AbstractLossFunction.h"
#include "../Solvers/AbstractSolver.h"

class AbstractGradientAlgo {


    /********************************************** ATTRIBUTES **************************************************/

friend class RnnModel;

public:

    /// Name of the Algorithm
    /**
     *  It is set during the construction of the object and cannot be modified later.
     */
    const std::string ALGO_NAME;


    /// Length of the processed sequences
    /**
     *  It is set during the construction of the object and cannot be modified later.
     */
    const unsigned SEQUENCE_LENGTH;


protected:

    /// Reference to the First Layer
    FirstLayer& dl1;

    /// Reference to the Second Layer
    SecondLayer& dl2;

    /// Pointer to Activation Function
    std::shared_ptr<AbstractActivationFunction> af1;

    /// Reference to Loss Function
    std::shared_ptr<AbstractLossFunction> loss;

    /// Collector for the previous outputs of the network
    /**
     */
    CircularBuffer<VectorClass> output_buffer;


    /********************************************** CONSTRUCTORS *********************************************/

protected:

    /// Constructor of the object.
    /**
     * @param dl1_:         First Layer of the network
     * @param dl2_:         Second Layer of the network
     * @param SEQ_LEN_:     Length of the sequences that will be processed (used for allocating memory)
     */
    AbstractGradientAlgo(std::string ALGO_NAME_,
                         FirstLayer& dl1_, SecondLayer& dl2_,
                         unsigned SEQUENCE_LENGTH_);

    /********************************************** METHODS *****************************************************/

protected:

  void set_dependencies(std::shared_ptr<AbstractActivationFunction> af1_ptr_,
      std::shared_ptr<AbstractLossFunction> loss_ptr_);

public:

    /// Process a sequence, computing output (forward propagation) and gradient of loss ("backpropagation")
    /**
     * @param x_sequence_train:     sequence of VectorClass object to be processed (exogenous inputs)
     * @param y_real:               target value
     */
    virtual void process_sequence(const std::vector<VectorClass>& x_sequence_train, const NN::type y_real) = 0;


    /// Compute output (forward propagation)
    /**
     * @param x_sequence_train:     sequence of VectorClass object to be processed (exogenous inputs)
     */
    void predict(const std::vector<VectorClass>& x_sequence_train);

};


#endif //RNN_P__LIBRARY_ABSTRACTGRADIENTALGO_H
