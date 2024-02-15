
#include "RnnModel.h"

#include <algorithm>


RnnModel::RnnModel(unsigned INPUT_NEURONS, unsigned HIDDEN_NEURONS,
        unsigned OUTPUT_NEURONS, const std::vector<unsigned>& WHICH_LAGS,
        unsigned DL1_SEED, unsigned DL2_SEED) :

        dl1(INPUT_NEURONS, HIDDEN_NEURONS, WHICH_LAGS, DL1_SEED),
        dl2(HIDDEN_NEURONS, OUTPUT_NEURONS, DL2_SEED),

        // default initialization
        PATIENCE(std::numeric_limits<unsigned>::infinity()),
        MIN_DELTA(0.),
        RESTORE_BEST(true),
        NUM_EPOCHS(100),
        BATCH_SIZE(32),

        rank(0),
        size(1) {}


//select activation function
void RnnModel::set_activation(const std::string& ACTIVATION_FUNCTION){

    if (ACTIVATION_FUNCTION == "ExpLU")
        af1_ptr = std::shared_ptr<AbstractActivationFunction>(new ExpLU(dl1.OUTPUT_SZ));
    else if (ACTIVATION_FUNCTION == "LeakyReLU")
        af1_ptr = std::shared_ptr<AbstractActivationFunction>(new LeakyReLU(dl1.OUTPUT_SZ));
    else if (ACTIVATION_FUNCTION == "Linear")
        af1_ptr = std::shared_ptr<AbstractActivationFunction>(new Linear(dl1.OUTPUT_SZ));
    else if (ACTIVATION_FUNCTION == "ReLU")
        af1_ptr = std::shared_ptr<AbstractActivationFunction>(new ReLU(dl1.OUTPUT_SZ));
    else if (ACTIVATION_FUNCTION == "Sigmoid")
        af1_ptr = std::shared_ptr<AbstractActivationFunction>(new Sigmoid(dl1.OUTPUT_SZ));
    else if (ACTIVATION_FUNCTION == "Softmax")
        af1_ptr = std::shared_ptr<AbstractActivationFunction>(new Softmax(dl1.OUTPUT_SZ));
    else if (ACTIVATION_FUNCTION == "Softsign")
        af1_ptr = std::shared_ptr<AbstractActivationFunction>(new Softsign(dl1.OUTPUT_SZ));
    else if (ACTIVATION_FUNCTION == "Swish")
        af1_ptr = std::shared_ptr<AbstractActivationFunction>(new Swish(dl1.OUTPUT_SZ));
    else if (ACTIVATION_FUNCTION == "Tanh")
        af1_ptr = std::shared_ptr<AbstractActivationFunction>(new Tanh(dl1.OUTPUT_SZ));
    else{
        std::cerr << "The selected activation function does not exist" << std::endl;
        exit(1);
    }

}


//select loss
void RnnModel::set_loss(const std::string& LOSS, NN::type LAMBDA){

    if (LOSS == "LogLikelihood")
        loss_ptr = std::shared_ptr<AbstractLossFunction>(new LogLikelihood(dl2.OUTPUT_SZ));
    else if (LOSS == "GaussianCRPS")
        loss_ptr = std::shared_ptr<AbstractLossFunction>(new GaussianCRPS(dl2.OUTPUT_SZ));
    else if (LOSS == "GaussianCRPSLambda")
        loss_ptr = std::shared_ptr<AbstractLossFunction>(new GaussianCRPSLambda(dl2.OUTPUT_SZ, LAMBDA));
    else if (LOSS == "GaussianCRPSTrivial")
        loss_ptr = std::shared_ptr<AbstractLossFunction>(new GaussianCRPSTrivial(dl2.OUTPUT_SZ, LAMBDA));
    else if (LOSS == "GaussianWinkler")
        loss_ptr = std::shared_ptr<AbstractLossFunction>(new GaussianWinkler(dl2.OUTPUT_SZ, LAMBDA));
    else if (LOSS == "GaussianWinklerTrivial")
        loss_ptr = std::shared_ptr<AbstractLossFunction>(new GaussianWinklerTrivial(dl2.OUTPUT_SZ, LAMBDA));
    else if (LOSS == "MSE")
        loss_ptr = std::shared_ptr<AbstractLossFunction>(new MSE(dl2.OUTPUT_SZ));
    else{
        std::cerr << "The selected loss does not exist" << std::endl;
        exit(1);
    }

}


//select metric
void RnnModel::set_metric(const std::string& METRIC){

    if (METRIC == "MSE_metric")
        metric_ptr = std::shared_ptr<AbstractMetric>(new MSE_metric(dl2.OUTPUT_SZ));
    else if (METRIC == "None")
        metric_ptr = nullptr;
    else{
        std::cerr << "The selected metric does not exist" << std::endl;
        exit(1);
    }

}


//select gradient algorithm
void RnnModel::set_gradientalgo(const std::string& ALGORITHM, unsigned SEQUENCE_LENGTH){

    if (ALGORITHM == "BPTT")
        gradient_algorithm_ptr = std::shared_ptr<AbstractGradientAlgo>(new BPTT(dl1, dl2, SEQUENCE_LENGTH));
    else if (ALGORITHM == "RTRL")
        gradient_algorithm_ptr = std::shared_ptr<AbstractGradientAlgo>(new RTRL(dl1, dl2, SEQUENCE_LENGTH));
    else if (ALGORITHM == "TRRL")
        gradient_algorithm_ptr = std::shared_ptr<AbstractGradientAlgo>(new TRRL(dl1, dl2, SEQUENCE_LENGTH));
    else{
        std::cerr << "The selected gradient algorithm does not exist" << std::endl;
        exit(1);
    }

}


//select optimizer
void RnnModel::set_optimizer(const std::string& SOLVER_, NN::type LEARNING_RATE_,
        unsigned NUM_EPOCHS_, unsigned BATCH_SIZE_, NN::type LR_DECAY_){

    BATCH_SIZE = BATCH_SIZE_;
    NUM_EPOCHS = NUM_EPOCHS_;

    if (SOLVER_ == "Adam") {
        optimizer_ptr = std::shared_ptr<AbstractSolver>(new Adam(dl1.NTP, dl2.NTP, LEARNING_RATE_, LR_DECAY_));
    }
    else if (SOLVER_ == "Nadam") {
        optimizer_ptr = std::shared_ptr<AbstractSolver>(new Nadam(dl1.NTP, dl2.NTP, LEARNING_RATE_, LR_DECAY_));
    }
    else {
        std::cerr << "The selected optimizer does not exist" << std::endl;
        exit(1);
    }

}


// add parameters for stopping criterion
void RnnModel::add_stopping_criterion(unsigned PATIENCE_, NN::type MIN_DELTA_, bool RESTORE_BEST_){

    PATIENCE = PATIENCE_;
    MIN_DELTA = MIN_DELTA_;
    RESTORE_BEST = RESTORE_BEST_;

}

/// Set paramters for MPI parallelization
void RnnModel::mpi_parallelize(int rank_, int size_){

    rank = rank_;
    size = size_;

}

// set default options for the training, if missing
void RnnModel::check_pointers() const{

    // check presence of activation function
    if (af1_ptr == nullptr){
        std::cerr << "Activation function is not set" << std::endl;
        exit(1);
    }

    // check presence of loss
    if (loss_ptr == nullptr){
        std::cerr << "Loss is not set" << std::endl;
        exit(1);
    }

    // check presence of metric
    if (metric_ptr == nullptr){
        std::cerr << "Warning: metric is not set" << "\n" << std::endl;
    }

    // check presence of gradient algorithm
    if (gradient_algorithm_ptr == nullptr){
        std::cerr << "Gradient algorithm is not set" << std::endl;
        exit(1);
    }

    // check presence of optimizer
    if (optimizer_ptr == nullptr){
        std::cerr << "Optimizer is not set" << std::endl;
        exit(1);
    }

}

void RnnModel::create_checkpoint(){

    dl1.save_status();
    dl2.save_status();

}

// Fit RNN Model
void RnnModel::fit(const std::vector< std::vector<VectorClass> >& x_sequences_train,
    const std::vector<NN::type>& y_sequences_train,
    std::ofstream& output_log, unsigned SHUFFLING_SEED){

    // check well-posedness of network
    check_pointers();

    // ensure dependencies of gradient algorithm are well defined
    gradient_algorithm_ptr->set_dependencies(af1_ptr, loss_ptr);

    // print header
    if (metric_ptr != nullptr)
        output_log << "\t" << "|||   Loss   |   Metric   |||" << std::endl;
    else
        output_log << "\t" << "|||   Loss   |||" << std::endl;

    // storing size for convenience
    const unsigned N_SEQUENCES_TRAIN = x_sequences_train.size();

    // define number of batches (rounding up to closest integer)
    const unsigned N_BATCHES = (N_SEQUENCES_TRAIN + BATCH_SIZE - 1) / BATCH_SIZE;

    // create a vector with all available indexes of training set
    std::vector<unsigned> INDEXES_TRAIN(N_SEQUENCES_TRAIN);
    std::iota(INDEXES_TRAIN.begin(), INDEXES_TRAIN.end(), 0);

    // set variables for stopping criterion
    unsigned epochs_since_new_min = 0;
    NN::type current_min = std::numeric_limits<NN::type>::infinity();

    // defining random number generator
    auto rng = std::default_random_engine(SHUFFLING_SEED);

    // training loop
    for (unsigned epoch_counter = 0; epoch_counter < NUM_EPOCHS && epochs_since_new_min < PATIENCE; ++epoch_counter){

        /***************************************** TRAINING *****************************************/

        // compute effective learning rate (useful only if a scheduled decay is present)
        optimizer_ptr->adjust_LR(epoch_counter);

        // shuffle indexes of the TRAINING SET
        //std::shuffle(INDEXES_TRAIN.begin(), INDEXES_TRAIN.end(), rng);

        // looping over the number of mini-batches
        for (unsigned batch = 0; batch < N_BATCHES; ++batch) {

            // index of the first element of the ‘batch-th’ mini-batch
            const unsigned STARTING = batch * BATCH_SIZE;
            const unsigned ELEMS_IN_BATCH = (batch != N_BATCHES-1) ? BATCH_SIZE : (N_SEQUENCES_TRAIN - STARTING);

            // initialize the batch gradients
            VectorClass dl1_local(dl1.NTP), dl2_local(dl2.NTP);

            // and the collector for MPI_Allreduce
            VectorClass dl1_aggregate(dl1.NTP), dl2_aggregate(dl2.NTP);

            for (unsigned elem_in_batch = rank; elem_in_batch < ELEMS_IN_BATCH; elem_in_batch += size) {

                unsigned my_index = STARTING + elem_in_batch;
                //std::cout << my_index << " " << INDEXES_TRAIN[ my_index ] << std::endl;

                //x_sequences_train[INDEXES_TRAIN[ my_index ]][0].print();
                //std::cout << y_sequences_train[INDEXES_TRAIN[ my_index ]] << "\n" <<std::endl;

                // then analyse forward and backward each sequence
                gradient_algorithm_ptr->process_sequence(
                        x_sequences_train[INDEXES_TRAIN[ my_index ]],
                        y_sequences_train[INDEXES_TRAIN[ my_index ]]);

                //dl1.gradient_params.print();
                //dl2.gradient_params.print();

                // update gradient of batch
                dl1_local += dl1.gradient_params;
                dl2_local += dl2.gradient_params;

                //dl1_local.print();
                //dl2_local.print();

            }

            // dividing by total number of elements (done before Allreducing, to avoid overflow)
            dl1_local /= ELEMS_IN_BATCH;
            dl2_local /= ELEMS_IN_BATCH;

            // Computing total sum of gradients through MPI_Allreduce
            //MPI_Allreduce(dl1_local.data(), dl1_aggregate.data(), dl1.NTP, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //7 feb
            //MPI_Allreduce(dl2_local.data(), dl2_aggregate.data(), dl2.NTP, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //7 feb

            // optimize
            //optimizer_ptr->solve(dl1_aggregate, dl2_aggregate); //7 feb
            optimizer_ptr->solve(dl1_local, dl2_local); // 7 feb (al posto di riga sopra)

            //dl1_local.print();
            //dl2_local.print();

            // update weights of the two layers
            dl1.new_weights( optimizer_ptr->dl1_weights_update );
            dl2.new_weights( optimizer_ptr->dl2_weights_update );

            dl1.print();
            dl2.print();

        }

        exit(1);
        /****************************************** ACCURACY ***************************************/

        // evaluating accuracy
        std::pair<NN::type, NN::type> epoch_avg_stats = evaluate_accuracy(x_sequences_train, y_sequences_train);

        // initializing stopping metric
        NN::type stopping_metric = 0.;

        // print results on the log file
        if (metric_ptr != nullptr){
            if (rank == 0){
                output_log << (epoch_counter+1) << " ||| "
                           << epoch_avg_stats.first << "  |  "
                           << epoch_avg_stats.second << " |||" << std::endl;
            }
            stopping_metric = epoch_avg_stats.second;
        }
        else{
            if (rank == 0){
                output_log << (epoch_counter+1) << " ||| "
                           << epoch_avg_stats.first << " |||" << std::endl;
            }
            stopping_metric = epoch_avg_stats.first;
        }

        // checking stopping criterion
        if (current_min - stopping_metric > MIN_DELTA) {
            current_min = stopping_metric;
            epochs_since_new_min = 0;
            create_checkpoint();
        }
        else {
            ++epochs_since_new_min;
        }

        std::cout << "Epoch " << epoch_counter << std::endl;
    }

    // restore best status for training
    if (RESTORE_BEST) {
        dl1.restore_status();
        dl2.restore_status();
    }

}


// Predict using RNN Model
std::vector<VectorClass> RnnModel::predict(const std::vector< std::vector<VectorClass> >& x_sequences){

    // storing size for convenience
    const unsigned N_SEQUENCES = x_sequences.size();

    // initializing predictions
    std::vector<VectorClass> predictions;
    predictions.reserve(N_SEQUENCES);

    // generate output and store predictions
    for (unsigned seq = 0; seq < N_SEQUENCES; ++seq) {
        gradient_algorithm_ptr->predict(x_sequences[seq]);
        predictions.push_back(dl2.output);
    }

    return predictions;

}


// Predict using RNN Model and compute stats
std::pair<NN::type, NN::type> RnnModel::evaluate_accuracy(const std::vector< std::vector<VectorClass> >& x_sequences,
    const std::vector<NN::type>& y_sequences){

    // storing size for convenience
    const unsigned N_SEQUENCES = x_sequences.size();

    // initialize variables for evaluating accuracy
    NN::type avg_loss_local = 0, avg_metric_local = 0;

    // and for MPI_Allreduce
    NN::type avg_loss_aggregate = 0, avg_metric_aggregate = 0;

    // analyse the accuracy of TRAINING SET
    for (unsigned seq = rank; seq < N_SEQUENCES; seq += size) {

        // generate prediction
        gradient_algorithm_ptr->predict(x_sequences[ seq ]);

        // compute loss and update stats
        loss_ptr->update_output(dl2.output, y_sequences[ seq ]);
        avg_loss_local += (loss_ptr->output / N_SEQUENCES);

        // compute metric and update stats, if required
        if (metric_ptr != nullptr){
            metric_ptr->update_output(dl2.output, y_sequences[ seq ]);
            avg_metric_local += (metric_ptr->output / N_SEQUENCES);
        }

    }

    // Computing total sum of loss and metric through MPI_Allreduce
    //MPI_Allreduce(&avg_loss_local, &avg_loss_aggregate, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // 7feb
    //MPI_Allreduce(&avg_metric_local, &avg_metric_aggregate, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // 7feb

    // return std::make_pair(avg_loss_aggregate, avg_metric_aggregate); 7feb
    return std::make_pair(avg_loss_local, avg_metric_local); // 7feb (al posto di riga sopra)

}
