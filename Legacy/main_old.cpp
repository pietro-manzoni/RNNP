#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>

#include <cstdio>
#include <random>
#include <assert.h>

#include "global.h"
#include "utilities.h"

#include "DataStructures/AlgebraicOperations.h"
#include "DataStructures/CircularBuffer.h"
#include "DataStructures/Matrix.h"
#include "DataStructures/VectorClass.h"
#include "DataStructures/Dataset.h"

#include "GradientAlgorithms/AbstractGradientAlgo.h"
#include "GradientAlgorithms/BPTT.h"
#include "GradientAlgorithms/RTRL.h"
#include "GradientAlgorithms/TRRL.h"

#include "Layers/FirstLayer.h"
#include "Layers/SecondLayer.h"

#include "Layers/ActivationFunctions/AbstractActivationFunction.h"
#include "Layers/ActivationFunctions/ExpLU.h"
#include "Layers/ActivationFunctions/Linear.h"
#include "Layers/ActivationFunctions/LeakyReLU.h"
#include "Layers/ActivationFunctions/ReLU.h"
#include "Layers/ActivationFunctions/Sigmoid.h"
#include "Layers/ActivationFunctions/Softmax.h"
#include "Layers/ActivationFunctions/Softsign.h"
#include "Layers/ActivationFunctions/Swish.h"
#include "Layers/ActivationFunctions/Tanh.h"

#include "Losses/AbstractLossFunction.h"
#include "Losses/LogLikelihood.h"
#include "Losses/MSE.h"

#include "Metrics/AbstractMetric.h"
#include "Metrics/MSE_metric.h"

#include "Models/LinearModel.h"
#include "Models/RnnModel.h"

#include "Solvers/Adam.h"
#include "Solvers/Nadam.h"



int main() {

    /**************************************************************************************************/
    /*************************************   DESIGNING the MODEL   ************************************/
    /**************************************************************************************************/

    /*
     * The design phase is structured as follows:
     *  1) Initial Settings:
     *  2) Output Format Settings:
     *  3) Data Importation:
     */

    /******************************************** INITIAL SETTINGS *******************************************/

    // Define name of the dataset
    std::string filename_ = "Data/final_dataset.csv";

    // Define directory name for output files (make sure it exists! the program does not check it...)
    std::string foldername_ = "OutFiles";

    // Set the number of decimal digits to be shown (Terminal). It is a global variable.
    NN::TRAINING_LOG_DIGITS = 10;

    // Set the number of decimal digits while printing fitted values/predictions. It is a global variable.
    NN::PREDICTION_DIGITS = 5;

    /*************************************** IMPORTING DATA *********************************************/

    // Import training and test set, keeping just relevant rows
    //    -> datetime format: "YYYY-MM-DD hh:mm:ss"
    // Here, we ignore returned indexes (i.e. the [numeric] position of the elements
    // in the dataframe that were kept), since we are not interested in them now
    Dataset dt_train(filename_);
    std::ignore = dt_train.reduce_dataset(">=", "2006-12-30 01:00:00");
    std::ignore = dt_train.reduce_dataset("<",  "2011-01-01 00:00:00");

    Dataset dt_test(filename_);
    std::ignore = dt_test.reduce_dataset(">=", "2010-12-30 00:00:00");
    std::ignore = dt_test.reduce_dataset("<",  "2012-01-01 00:00:00");

    /*******************************  REGRESSORS and REGRESSANDS  ***********************************/

    // Define column of exogenous inputs
    const std::string REGRESSAND_VARIABLE = "Demand";

    // Define columns of exogenous inputs for GLM
    const std::vector<std::string> GLM_INPUT_COLUMNS = {
            "Intercept",
            "SY1", "CY1", "SY2", "CY2",
            "Trend",
            "DoW_5","DoW_6", "Holiday"   //(Sat,Sun,Hol)
    };

    // Define columns of exogenous inputs for NN
    const std::vector<std::string> NN_INPUT_COLUMNS = {
            "Drybulb", "Dewpnt",

            "DoW_0", "DoW_1", "DoW_2", "DoW_3","DoW_4",     //(Mon,...,Fri)
            "DoW_5","DoW_6",  "Holiday",                    //(Sat,Sun,Hol)

            "SY1", "CY1", "SY2", "CY2",
            "SD1", "CD1", "SD2", "CD2"

    };

    /***********************************  RNN SETTINGS  ************************************/

    // the size of NN exogenous inputs is stored
    const unsigned INPUT_NEURONS = NN_INPUT_COLUMNS.size(); //TODO: fra un po', lo spostiamo in basso

    // Network Hyper-parameters

    // set whether to use Point (1 output node) or Probabilistic (2 output nodes) Forecasting
    NN::OUTPUT_NODES = 2;

    // Network Topology
    const unsigned HIDDEN_NEURONS = 10;                    // neurons in the first hidden layer
    const std::vector<unsigned> WHICH_LAGS = {1,2,24};      // which lags to use as feedback

    // Training and optimization
    const NN::type LEARNING_RATE = 0.001;                   // learning rate
    const unsigned NUM_EPOCHS = 1000;                       // maximum number of epochs for training
    const unsigned BATCH_SIZE = 32;                         // number of sub-sequences in each mini-batch
    const NN::type LR_DECAY = 0.0;                          // decay of learning rate

    // Regularization
    const NN::type L1_REGULARIZATION = 0.0;                 // L1 regularization
    const NN::type L2_REGULARIZATION = 0.0;                 // L2 regularization

    // Seeds for first and second layer and shuffling random engine
    const unsigned DL1_SEED = 100;
    const unsigned DL2_SEED = 200;
    const unsigned SHUFFLING_SEED = 10;

    // Define of Training Set features
    const unsigned SEQUENCE_LENGTH = 48;        // length of each sequence (fed into the RNN)
    const unsigned WINDOW_SHIFT = 1;            // distance between the beginning of two consecutive sequences

    // Stopping criterion
    const unsigned PATIENCE = 100;
    const NN::type MIN_DELTA = 0;

    // Build the Network, defining the Jordan's feedbacks
    FirstLayer dl1(INPUT_NEURONS, HIDDEN_NEURONS, WHICH_LAGS, DL1_SEED);
    Sigmoid af1(HIDDEN_NEURONS);
    SecondLayer dl2(HIDDEN_NEURONS, NN::OUTPUT_NODES, DL2_SEED);

    // Select Solver
    Adam optimizer(dl1, dl2, LEARNING_RATE, LR_DECAY);

    // Select  Loss and Metric
    //MSE loss(NN::OUTPUT_NODES);          // Point case
    LogLikelihood loss(NN::OUTPUT_NODES);   // Probabilistic case
    MSE_metric metric(NN::OUTPUT_NODES);

    // Algorithm for Gradient Computation
    TRRL gradient_algorithm(dl1, af1, dl2, loss, SEQUENCE_LENGTH);


    /**************************************************************************************************/
    /*******************************************   CHECKS   *******************************************/
    /**************************************************************************************************/

    assert("Unsupported output size" &&
        (NN::OUTPUT_NODES == 1 || NN::OUTPUT_NODES == 2 ) );

    assert("Incompatible loss and output size" &&
        ((NN::OUTPUT_NODES == 1 && loss.LOSS_NAME == "MSE" ) ||
         (NN::OUTPUT_NODES == 2 && loss.LOSS_NAME == "LogLikelihood" )) );


    /**************************************************************************************************/
    /************************************ SETTING OUTPUT FORMAT ***************************************/
    /**************************************************************************************************/

    // set the number of digits to be shown.
    std::cout << std::setprecision(NN::TRAINING_LOG_DIGITS) << std::fixed;

    // open the output stream and set the number precision
    std::ofstream output_log(foldername_ + "/training_stats.csv");
    output_log << std::setprecision(NN::TRAINING_LOG_DIGITS) << std::fixed;


    /**************************************************************************************************/
    /*************************************   TRAINING the MODEL   *************************************/
    /**************************************************************************************************/
    /*
     * The training phase is composed by the following operations:
     * 1) Importation of Datasets
     *
     */


    /************************ IMPORTING DATASETS and NORMALIZING  ***************************/

    // store the (observed!) target values for convenience
    std::vector<NN::type> target_train = dt_train.extract_vector(REGRESSAND_VARIABLE);
    std::vector<NN::type> target_test = dt_test.extract_vector(REGRESSAND_VARIABLE);

    // store the corresponding dates
    std::vector<std::string> dates_train = dt_train.get_row_index();
    std::vector<std::string> dates_test = dt_test.get_row_index();

    // apply logarithmic transformation to target variable (consumption)
    dt_train.apply_log(REGRESSAND_VARIABLE);
    dt_test.apply_log(REGRESSAND_VARIABLE);

    // minmax normalize the dataframes (avoid normalizing "Hour" and "Intercept")
    dt_train.normalize_dataframe( {"Hour", "Intercept"} );
    dt_test.normalize_dataframe( dt_train.get_minmax() );


    /***************************** GENERAL LINEAR MODEL  ***********************************/

    // creating vectors for collecting the Linear part of the hybrid model
    std::vector<NN::type> GLM_fitted(dt_train.get_nrows());
    std::vector<NN::type> GLM_predicted(dt_test.get_nrows());

    // looping over the 24 hours (for removing macroscopic seasonality)
    for (unsigned hh = 0; hh < 24; ++hh){

        // create copies of the datasets that will be modified
        Dataset dt_train_GLM = dt_train;
        Dataset dt_test_GLM = dt_test;

        // reducing the new datasets (and store the survived indexes)
        auto glm_idx_train = dt_train_GLM.reduce_dataset("Hour", "==", hh);
        auto glm_idx_test  = dt_test_GLM.reduce_dataset("Hour", "==", hh);

        // extracting features
        auto x_train_GLM = dt_train_GLM.extract_matrix(GLM_INPUT_COLUMNS);
        auto x_test_GLM = dt_test_GLM.extract_matrix(GLM_INPUT_COLUMNS);

        // extracting features
        std::vector<NN::type> y_train_GLM = dt_train_GLM.extract_vector(REGRESSAND_VARIABLE);
        std::vector<NN::type> y_test_GLM = dt_test_GLM.extract_vector(REGRESSAND_VARIABLE);

        // creating and fitting the LinearModel object
        LinearModel lm;
        lm.fit(x_train_GLM, y_train_GLM);

        //extracting predictions
        std::vector<NN::type> tmp_fitted = lm.predict(x_train_GLM);
        std::vector<NN::type> tmp_predicted = lm.predict(x_test_GLM);

        //inserting the forecasts in the full vectors, using stored indexes
        for (unsigned i = 0; i < glm_idx_train.size(); ++i)
            GLM_fitted[ glm_idx_train[i] ] = tmp_fitted[i];
        for (unsigned i = 0; i < glm_idx_test.size(); ++i)
            GLM_predicted[ glm_idx_test[i] ] = tmp_predicted[i];

    }

    /**********************************  CREATE RNN V2  ************************************/

    // the size of NN exogenous inputs is stored
    // const unsigned INPUT_NEURONS = NN_INPUT_COLUMNS.size();

    // Initializing Neural Network
    RnnModel rnn(INPUT_NEURONS, HIDDEN_NEURONS, NN::OUTPUT_NODES,
                 WHICH_LAGS, DL1_SEED, DL2_SEED);

    // set activation function for hidden states
    rnn.set_activation("Sigmoid");

    // set algorithm for computation of gradients
    rnn.set_gradientalgo("TRRL");

    // set optimizer and related hyperparameters
    rnn.set_optimizer("Adam", LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, LR_DECAY);

    // add a stopping criterion
    rnn.add_stopping_criterion(PATIENCE, MIN_DELTA);

    // finally, metric and loss are left as default


    /************************* EXTRACTING RNN INPUTS and TARGETS  *******************************/

    // selected columns are extracted and stored
    std::vector<VectorClass> x_train_NN = dt_train.extract_matrix(NN_INPUT_COLUMNS);
    std::vector<VectorClass> x_test_NN = dt_test.extract_matrix(NN_INPUT_COLUMNS);

    // extract regressand variable
    std::vector<NN::type> residual_train = dt_train.extract_vector(REGRESSAND_VARIABLE);
    std::vector<NN::type> residual_test = dt_test.extract_vector(REGRESSAND_VARIABLE);

    // and remove the GLM part
    for (unsigned i = 0; i < residual_train.size(); ++i)
        residual_train[i] -= GLM_fitted[i];
    for (unsigned i = 0; i < residual_test.size(); ++i)
        residual_test[i] -= GLM_predicted[i];

    // print some relevant info
    std::cout << "\n" << "Training set [fitted values]:" << "\n"
              << "- first datetime: " << dt_train.get_row_index().at(SEQUENCE_LENGTH-1)  << "\n"
              << "- last datetime: "  << dt_train.get_row_index().back() << "\n"
              << "- number of elements: " << dt_train.get_nrows() << std::endl;
    std::cout << "Test set [predicted values]:" << "\n"
              << "- first datetime: " << dt_test.get_row_index().at(SEQUENCE_LENGTH-1)  << "\n"
              << "- last datetime: "  << dt_test.get_row_index().back() << "\n"
              << "- number of elements: " << dt_test.get_nrows() << "\n" << std::endl;

    /************************ CREATING SEQUENCES for FEEDING RNN  ******************************/

    // initialize structures
    std::vector< std::vector<VectorClass> > x_sequences_train, x_sequences_test;
    std::vector<NN::type> y_sequences_train, y_sequences_test;

    // creating the subsequences (of length SEQUENCE_LENGTH and with step WINDOW_SHIFT)
    split_dataset(x_train_NN, residual_train, x_sequences_train, y_sequences_train, SEQUENCE_LENGTH, WINDOW_SHIFT);
    split_dataset(x_test_NN, residual_test, x_sequences_test, y_sequences_test, SEQUENCE_LENGTH, WINDOW_SHIFT);

    // storing sizes for convenience
    const unsigned N_SEQUENCES_TRAIN = x_sequences_train.size();
    const unsigned N_SEQUENCES_TEST = x_sequences_test.size();

    // define the number of “complete” batches
    const unsigned N_FULL_BATCHES = N_SEQUENCES_TRAIN / BATCH_SIZE;

    // create a vector with all available indexes of training set
    std::vector<unsigned> INDEXES_TRAIN(N_SEQUENCES_TRAIN);
    std::iota(INDEXES_TRAIN.begin(), INDEXES_TRAIN.end(), 0);

    // set variables for early stopping
    unsigned epochs_since_new_min = 0;
    NN::type current_min = std::numeric_limits<NN::type>::infinity();

    // defining random number generator
    auto rng = std::default_random_engine(SHUFFLING_SEED);

    // tic: start measuring training time
    auto t1 = std::chrono::high_resolution_clock::now();

    // header for training stats
    std::cout  << "\t" << "|||   Loss   |   Metric   |||" << std::endl;
    output_log << "\t" << "|||   Loss   |   Metric   |||" << std::endl;

    // training loop
    for (unsigned epoch_counter = 1; epoch_counter <= NUM_EPOCHS && epochs_since_new_min < PATIENCE; ++epoch_counter) {

        // shuffle indexes of the TRAINING SET
        std::shuffle(INDEXES_TRAIN.begin(), INDEXES_TRAIN.end(), rng);

        /******************************************* EPOCH ****************************************/

        // looping over the number of full mini-batches
        for (unsigned batch = 0; batch < N_FULL_BATCHES; ++batch) {

            // index of the first element of the ‘batch-th’ mini-batch
            unsigned starting = batch * BATCH_SIZE;

            for (unsigned iteration_in_batch = 0; iteration_in_batch < BATCH_SIZE; ++iteration_in_batch) {

                // then analyse forward and backward each sequence
                gradient_algorithm.process_sequence(
                        x_sequences_train[INDEXES_TRAIN[starting + iteration_in_batch]],
                        y_sequences_train[INDEXES_TRAIN[starting + iteration_in_batch]]);

                // optimize
                const bool EOB = (iteration_in_batch == BATCH_SIZE - 1); //END-OF-BATCH
                optimizer.solve(EOB, epoch_counter);

            }

        }

        /******************************************* ACCURACY ****************************************/

        // initialize variables for evaluating accuracy
        NN::type epoch_avg_loss_train = 0, epoch_avg_metric_train = 0;

        // analyse the accuracy of TRAINING SET
        for (unsigned seq = 0; seq < N_SEQUENCES_TRAIN; ++seq) {

            // predict output (it is stored in dl2.output)
            gradient_algorithm.predict(x_sequences_train[seq]);

            // compute loss and metric
            loss.update_output(dl2.output, y_sequences_train[seq]);
            metric.update_output(dl2.output, y_sequences_train[seq]);

            // UPDATE STATS
            epoch_avg_loss_train += (loss.output / N_SEQUENCES_TRAIN);
            epoch_avg_metric_train += (metric.output / N_SEQUENCES_TRAIN); //MSE

        }

        /******************************************* PRINT STATS ****************************************/

        // print the results for this EPOCH
        std::cout << epoch_counter << " ||| " <<
                  epoch_avg_loss_train << "  |  " << epoch_avg_metric_train << " |||" << std::endl;

        // print results on the log file
        output_log << epoch_counter << " ||| " <<
                      epoch_avg_loss_train << "  |  " << epoch_avg_metric_train << " |||" << std::endl;

        // Early stopping criterion (on training set)
        if (current_min - epoch_avg_metric_train > MIN_DELTA) {
            current_min = epoch_avg_metric_train;
            epochs_since_new_min = 0;
            create_checkpoint(dl1, dl2);
        } else
            ++epochs_since_new_min;

        // at the end of this loop, the entire training set has been processed once (one epoch)
    }

    // toc: print the elapsed time for the NN training
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    // print elapsed time
    output_log << "\n" << "Elapsed time [RNN training]: " << duration << " seconds" << std::endl;


    /******************************************* EXPORT PREDICTION ****************************************/

    if (true) { // if rank is 0

        // restoring best status of trained layers
        dl1.restore_status();
        dl2.restore_status();

        /******************************** INITIALIZING and ADJUSTING **************************************/

        // declaration
        std::vector<NN::type> mean_fitted(N_SEQUENCES_TRAIN);
        std::vector<NN::type> stdev_fitted(N_SEQUENCES_TRAIN);
        std::vector<NN::type> point_fitted(N_SEQUENCES_TRAIN);

        std::vector<NN::type> mean_predicted(N_SEQUENCES_TEST);
        std::vector<NN::type> stdev_predicted(N_SEQUENCES_TEST);
        std::vector<NN::type> point_predicted(N_SEQUENCES_TEST);

        // necessary for de-normalization
        const auto MINMAX = dt_train.get_minmax(REGRESSAND_VARIABLE);

        // aligning vectors (discarding first "SEQUENCE_LENGTH-1" elements)
        GLM_fitted.erase(GLM_fitted.begin(), GLM_fitted.begin() + SEQUENCE_LENGTH - 1);
        GLM_predicted.erase(GLM_predicted.begin(), GLM_predicted.begin() + SEQUENCE_LENGTH - 1);

        target_train.erase(target_train.begin(), target_train.begin() + SEQUENCE_LENGTH - 1);
        target_test.erase(target_test.begin(), target_test.begin() + SEQUENCE_LENGTH - 1);

        dates_train.erase(dates_train.begin(), dates_train.begin() + SEQUENCE_LENGTH - 1);
        dates_test.erase(dates_test.begin(), dates_test.begin() + SEQUENCE_LENGTH - 1);


        /********************************************* PRINTING *******************************************/

        // open output streams
        std::ofstream outstreamFitted(foldername_ + "/fitted.csv");
        outstreamFitted << std::setprecision(NN::PREDICTION_DIGITS) << std::fixed;
        outstreamFitted << "Date," << REGRESSAND_VARIABLE <<
            ",Forecast,GLM,Mu,Sigma" << std::endl;

        std::ofstream outstreamPredicted(foldername_ + "/predicted.csv");
        outstreamPredicted << std::setprecision(NN::PREDICTION_DIGITS) << std::fixed;
        outstreamPredicted << "Date," << REGRESSAND_VARIABLE <<
            ",Forecast,GLM,Mu,Sigma" << std::endl;

        // process training set
        for (unsigned seq = 0; seq < N_SEQUENCES_TRAIN; ++seq) {

            // predict output (it is stored in dl2.output)
            gradient_algorithm.predict(x_sequences_train[seq]);

            // export predicted values
            NN::type glm = GLM_fitted[seq];
            NN::type mu = dl2.output(0);
            NN::type sigma = (NN::OUTPUT_NODES==2) ? fabs(dl2.output(1)) : 0.0;

            // denormalizing (minmax)
            glm = glm * (MINMAX.second - MINMAX.first) + MINMAX.first;
            mu  = mu  * (MINMAX.second - MINMAX.first);
            sigma = sigma * (MINMAX.second - MINMAX.first);

            mean_fitted[seq] = glm + mu;
            stdev_fitted[seq] = sigma;
            point_fitted[seq] = exp(mean_fitted[seq] + 0.5 * stdev_fitted[seq] * stdev_fitted[seq]);

            outstreamFitted << dates_train[seq] << ","
                            << target_train[seq] << ","
                            << point_fitted[seq] << ","
                            << glm << ","
                            << mean_fitted[seq] << ","
                            << stdev_fitted[seq] << std::endl;
        }

        // process testing set
        for (unsigned seq = 0; seq < N_SEQUENCES_TEST; ++seq) {

            // predict output (it is stored in dl2.output)
            gradient_algorithm.predict(x_sequences_test[seq]);

            // export predicted values
            NN::type glm = GLM_predicted[seq];
            NN::type mu = dl2.output(0);
            NN::type sigma = (NN::OUTPUT_NODES==2) ? fabs(dl2.output(1)) : 0.0;

            // denormalizing (minmax)
            glm = glm * (MINMAX.second - MINMAX.first) + MINMAX.first;
            mu  = mu  * (MINMAX.second - MINMAX.first);
            sigma = sigma * (MINMAX.second - MINMAX.first);

            mean_predicted[seq] = glm + mu;
            stdev_predicted[seq] = sigma;
            point_predicted[seq] = exp(mean_predicted[seq] + 0.5 * stdev_predicted[seq] * stdev_predicted[seq]);

            outstreamPredicted  << dates_test[seq]  << ","
                                << target_test[seq] << ","
                                << point_predicted[seq] << ","
                                << glm << ","
                                << mean_predicted[seq] << ","
                                << stdev_predicted[seq] << std::endl;

        }

        /********************************************* STATISTICS *******************************************/

        output_log << "\n" << "Training Set:" << std::endl;
        compute_statistics(mean_fitted, stdev_fitted, point_fitted,
                           target_train, output_log);

        output_log << "\n" << "Test Set:" << std::endl;
        compute_statistics(mean_predicted, stdev_predicted, point_predicted,
                           target_test, output_log);

    }

    return 0;

}
