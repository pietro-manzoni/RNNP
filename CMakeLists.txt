cmake_minimum_required(VERSION 3.21)
project(RNNOverconfidence)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)

add_executable(

        # executable name
        RNNOverconfidence

        #main
        main_grid_search_Gigat.cpp

        #global variables
        global.cpp
        global.h

        #Activation Functions
        ActivationFunctions/AbstractActivationFunction.cpp
        ActivationFunctions/AbstractActivationFunction.h
        ActivationFunctions/ExpLU.cpp
        ActivationFunctions/ExpLU.h
        ActivationFunctions/LeakyReLU.cpp
        ActivationFunctions/LeakyReLU.h
        ActivationFunctions/Linear.cpp
        ActivationFunctions/Linear.h
        ActivationFunctions/ReLU.cpp
        ActivationFunctions/ReLU.h
        ActivationFunctions/Sigmoid.cpp
        ActivationFunctions/Sigmoid.h
        ActivationFunctions/Softmax.cpp
        ActivationFunctions/Softmax.h
        ActivationFunctions/Softsign.cpp
        ActivationFunctions/Softsign.h
        ActivationFunctions/Swish.cpp
        ActivationFunctions/Swish.h
        ActivationFunctions/Tanh.cpp
        ActivationFunctions/Tanh.h

        #DataStructures
        DataStructures/AlgebraicOperations.cpp
        DataStructures/AlgebraicOperations.h
        DataStructures/CircularBuffer.h
        DataStructures/Dataset.cpp
        DataStructures/Dataset.h
        DataStructures/Matrix.cpp
        DataStructures/Matrix.h
        DataStructures/VectorClass.cpp
        DataStructures/VectorClass.h

        # Gradient Algorithms
        GradientAlgorithms/AbstractGradientAlgo.cpp
        GradientAlgorithms/AbstractGradientAlgo.h
        GradientAlgorithms/BPTT.cpp
        GradientAlgorithms/BPTT.h
        GradientAlgorithms/TRRL.cpp
        GradientAlgorithms/TRRL.h
        GradientAlgorithms/RTRL.cpp
        GradientAlgorithms/RTRL.h

        #Dense Layers
        Layers/FirstLayer.cpp
        Layers/FirstLayer.h
        Layers/SecondLayer.h

        # Losses
        Losses/AbstractLossFunction.cpp
        Losses/AbstractLossFunction.h
        Losses/GaussianCRPS.cpp
        Losses/GaussianCRPS.h
        Losses/GaussianCRPSLambda.cpp
        Losses/GaussianCRPSLambda.h
        Losses/GaussianCRPSTrivial.cpp
        Losses/GaussianWinkler.h
        Losses/GaussianWinkler.cpp
        Losses/GaussianWinklerTrivial.h
        Losses/GaussianWinklerTrivial.cpp
        Losses/LogLikelihood.cpp
        Losses/LogLikelihood.h
        Losses/MSE.cpp
        Losses/MSE.h

        #Metrics
        Metrics/AbstractMetric.cpp
        Metrics/AbstractMetric.h
        Metrics/MSE_metric.cpp
        Metrics/MSE_metric.h

        #Models
        Models/LinearModel.cpp
        Models/LinearModel.h
        Models/RNNModel.cpp
        Models/RNNModel.h

        #Solvers
        Solvers/AbstractSolver.cpp
        Solvers/AbstractSolver.H
        Solvers/Adam.cpp
        Solvers/Adam.h
        Solvers/Nadam.cpp
        Solvers/Nadam.h

        #utilities
        utilities.cpp
        utilities.h

        )

target_link_libraries(RNNOverconfidence PUBLIC MPI::MPI_CXX)