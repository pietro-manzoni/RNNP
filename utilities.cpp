
#include <numeric>
#include <iomanip>
#include <cmath>

#include "utilities.h"


void split_dataset(const std::vector<VectorClass>& x_dataset,
                   const std::vector<NN::type>& y_dataset,
                   std::vector< std::vector<VectorClass> >& x_sequences,
                   std::vector<NN::type>& y_sequences,
                   const unsigned WINDOW_SIZE, const unsigned TIME_DISTANCE){

    // store number of rows
    const unsigned DATASET_LEN = x_dataset.size();

    // create subsequences
    for (unsigned n = 0; n < DATASET_LEN - WINDOW_SIZE + 1; n += TIME_DISTANCE){

        // instantiate containers that will collect each subsequence (of exogenous inputs)
        std::vector<VectorClass> x_tmp;

        // copying values into the matrix and vector
        for (unsigned i = 0; i < WINDOW_SIZE; ++i) {
            x_tmp.push_back( x_dataset[n+i] );
        }

        // storing exogenous sequences and the target value
        x_sequences.push_back( x_tmp );
        y_sequences.push_back( y_dataset[n+WINDOW_SIZE-1] );

    }

}


NN::type compute_quantile(NN::type mu, NN::type sigma, NN::type alpha){

    // compute quantile of level alpha (with respect to standard normal)
    NN::type std_quantile = norminv(alpha);

    // return EXPONENTIATED alpha quantile
    return exp(mu + std_quantile * sigma);

}


std::pair<NN::type, NN::type> compute_ci(NN::type mu, NN::type sigma, NN::type alpha){

  // compute quantile of level alpha (with respect to standard normal)
  NN::type std_quantile = norminv(0.5 * (1+alpha));

  // return EXPONENTIATED alpha confidence interval
  return std::make_pair(exp(mu - std_quantile * sigma),
                        exp(mu + std_quantile * sigma));

}


NN::type compute_statistics(const std::vector<NN::type>& mu, const std::vector<NN::type>& sigma,
    const std::vector<NN::type>& point_forecast, const std::vector<NN::type>& consumption_true,
    std::ofstream& output_stats){

    // extracting number of sequences
    const unsigned N_SEQUENCES = consumption_true.size();

    // compute RMSE and MAPE
    NN::type RMSE = 0;
    for (unsigned i = 0; i < N_SEQUENCES; ++i) {
        RMSE += (consumption_true[i] - point_forecast[i]) * (consumption_true[i] - point_forecast[i]) / N_SEQUENCES;
    }
    RMSE = sqrt(RMSE);

    NN::type MAPE = 0;
    for (unsigned i = 0; i < N_SEQUENCES; ++i) {
        MAPE += (fabs(consumption_true[i] - point_forecast[i]) / consumption_true[i]) / N_SEQUENCES;
    }

    output_stats << "RMSE: " << RMSE << " | MAPE: " << MAPE << std::endl;

    // skip confidence intervals analysis
    if (NN::OUTPUT_NODES == 1)
      return NN::NaN;

    // PINBALL LOSS on PERCENTILES (1-99)
    NN::type APL = 0;
    for (unsigned percentile = 1; percentile < 100; ++percentile){

        NN::type percentile_score = 0.;
        NN::type alpha = static_cast<NN::type>(percentile) / 100.;

        for (unsigned i = 0; i < N_SEQUENCES; ++i) {
            auto quantiles = compute_quantile(mu[i], sigma[i], alpha);
            if (quantiles > consumption_true[i])
                percentile_score += (1-alpha) * (quantiles - consumption_true[i]);
            else
                percentile_score += alpha * (consumption_true[i] - quantiles);
        }
        APL += percentile_score / (99*N_SEQUENCES);
    }
    output_stats << "APL: " << APL << std::endl;

    // WINKLER SCORE on PERCENTILES (1-99)
    NN::type AWS = 0;
    for (unsigned percentile = 1; percentile < 100; ++percentile){

        NN::type percentile_score = 0.;
        NN::type alpha = static_cast<NN::type>(percentile) / 100.;

        for (unsigned i = 0; i < N_SEQUENCES; ++i) {
            auto upper_CI = compute_quantile(mu[i], sigma[i], (1+alpha)/2);
            auto lower_CI = compute_quantile(mu[i], sigma[i], (1-alpha)/2);
	    percentile_score += (upper_CI - lower_CI);
            if (lower_CI > consumption_true[i])
                percentile_score += 2/(1-alpha) * (lower_CI - consumption_true[i]);
            if (upper_CI < consumption_true[i])
                percentile_score += 2/(1-alpha) * (consumption_true[i] - upper_CI);
        }
        AWS += percentile_score / (99*N_SEQUENCES);
    }
    output_stats << "AWS: " << AWS << std::endl;



    // BACKTESTING

    // initialise collector for backtesting 95%
    NN::type b95 = 0.0;

    for (unsigned level = 90; level < 100; ++level){

        unsigned exceptions = 0;
        NN::type alpha = static_cast<NN::type>(level) / 100.;

        for (unsigned i = 0; i < N_SEQUENCES; ++i) {
            auto CI = compute_ci(mu[i], sigma[i], alpha);
            if ((consumption_true[i] < CI.first) || (consumption_true[i] > CI.second))
                ++exceptions;
        }

        NN::type realized_level = 1. -
            static_cast<NN::type>(exceptions) / static_cast<NN::type>(N_SEQUENCES);
        output_stats << "Backtesting " << level << " %: " << realized_level << std::endl;

        if (level == 95)
            b95 = realized_level;

    }

    return b95;

}


NN::type normpdf(NN::type x){
    return 1 / sqrt(2*M_PI) * exp(-0.5*x*x);
}


NN::type normcdf(NN::type x){
    return 0.5 * erfc(-x * sqrt(0.5));
}


NN::type norminv(NN::type alpha, NN::type tol, unsigned max_iter){

    // consistency check
    if (alpha == 0.)
        return - NN::infty;
    else if (alpha == 0.)
        return - NN::infty;
    else if(alpha < 0. || alpha > 1.){
        std::cerr << "Invalid alpha argument provided to norminv" << std::endl;
        return NN::NaN;
    }

    // if alpha is ok: we use Polya (1945) approximation to find initial point
    // (cf. also "Approximations to Standard Normal Distribution Function", R. Yerukala and N.K. Boiroju 2015)
    NN::type x0 = 0;

    if (alpha >= 0.5)
        x0 = + sqrt( - M_PI/2 * log(4 * alpha * (1-alpha)) );
    else
        x0 = - sqrt( - M_PI/2 * log(4 * alpha * (1-alpha)) );

    // compute residual value
    NN::type resid = normcdf(x0) - alpha;

    // and use Newton-Rapson for inverting the normcdf
    unsigned n_iter = 0;
    while (fabs(resid) > tol && n_iter < max_iter ){
        x0 -= resid / normpdf(x0);    //point update
        resid = normcdf(x0) - alpha;  //update residual value
        ++n_iter;                     //increase counter for max_iter criterion
    }

    // if necessary, print warning
    if (n_iter == max_iter)
        std::cerr << "Warning: maximum number of iteration reached in norminv";

    return x0;

}

// returns the integral between 0 and t of the logarithmic Q-function
NN::type integratelogQ(NN::type t){

    if (t < 0){
        std::cerr << "logQ cannot be integrated: required positive parameter";
        return NN::NaN;
    }

    // values of the integrals on integer values, as obtained using Matlab
    // quadgk function with options  ('RelTol', 1e-9, 'MaxIntervalCount', 1e9)
    // to integrate log(normcdf(x)) between -t and 0
    std::vector<NN::type> numeric_integral{
          0.,             -1.206338229,    -3.947688598,   // 0,1,2
         -9.067282064,   -17.472634874,   -30.105100419,   // 3,4,5
        -47.924980541,   -71.903927059,  -103.020812252,   // 6,7,8
       -142.259341227,  -190.606593346,  -249.052088072,   // 9,10,11
       -318.587162111,  -400.204539596,  -494.898026636,   // 12,13,14
       -603.662288658,  -727.492684471,  -867.385140201,   // 15,16,17
      -1024.336051862, -1199.342208908, -1393.400733447};  // 18,19,20

    // Quintics approximations in the interval (n, n+1). Computed using Matlab
    // fmincon and minimizing the total squared error with a 0.0001 step:
    // q = c(0) + c(1) * x.^1 + c(2) * x.^2 + c(3) * x.^3 + c(4) * x.^4 + c(5) * x.^5;
    std::vector<std::vector<NN::type>> quintics{
      { -0.693146964607974,  0.797889700116101, -0.318269821578586,  0.036489060676123,  0.005103294757339,  0.000329170962613}, //(0,1)
      { -0.699367383894136,  0.778424327066362, -0.342133751174458,  0.022216878847683,  0.000996661804519, -0.000104708763261}, //(1,2)
      { -0.723580594102934,  0.758486380287559, -0.334100389927477,  0.035601331200738,  0.005849273432556,  0.000470490099714}, //(2,3)
      { -0.577481744045063,  1.031532487177306, -0.147205553945127,  0.094811795583944,  0.014447415492475,  0.000909969980669}, //(3,4)
      { -0.243950386070573,  0.846581553479763, -0.505045298837493, -0.054721210853915, -0.011281113111684, -0.000719567328273}, //(4,5)
      { -0.001242753133633,  1.123610705689695, -0.315087041878941,  0.012291971391883, -0.000322732652468, -0.000054341857127}, //(5,6)
      {  0.028146304044870,  0.960891590125525, -0.391086647071657,  0.001568268655428, -0.000676419670466, -0.000038008727626}, //(6,7)
      {  5.116719315784466,  3.626048032010075,  0.043264454318453,  0.011800445948633, -0.003886566448200, -0.000246305396781}, //(7,8)
      { -0.924936136762239,  2.413929734229947,  0.153069838797084,  0.059648058641086,  0.000778668364336, -0.000084621363717}, //(8,9)
      {  1.042307838004707, -0.656100346755728, -0.909985305680899, -0.042181242387998, -0.000789180670669,  0.000041256341810}, //(9,10)
      {  2.932822789398055, -0.936048554911733, -1.016387524793171, -0.047537885477859, -0.000717523251525,  0.000042460845896}, //(10,11)
      { -0.216174394288872,  1.712444560226468, -0.169899933606145,  0.025251487797613,  0.000563366583926, -0.000007134353679}, //(11,12)
      {-24.173948248543439, -3.454163187543578, -0.523405939088699,  0.018975925753200,  0.000697525628935, -0.000004051074708}, //(12,13)
      { -0.025930385639900,  1.430636284053134, -0.301676228107170,  0.011043179557425,  0.000202312849982, -0.000000268793308}, //(13,14)
      { -2.754623818576503,  1.778570771803463, -0.099881556972172,  0.033293715958917,  0.001144007427801,  0.000012870473546}, //(14,15)
      {  3.311900803955689,  2.842899482696244, -0.207164126690815,  0.001196581858360, -0.001035397620105, -0.000034596188826}, //(15,16)
      { -2.103509605495522,  2.540922732476315, -0.056134698631108,  0.026378479286020,  0.000473181075976, -0.000002356878819}, //(16,17)
      { -4.759249056618198,  0.831316902878632, -0.107137443498026,  0.048481183209163,  0.002375039659175,  0.000041256729548}, //(17,18)
      {  8.349732025958678, -3.238159064176356, -1.181196111598166, -0.034441572615884, -0.000311434043679,  0.000009481509201}, //(18,19)
      { 14.788312641044529, -2.581348421628751, -1.060837140512429, -0.023970035695996, -0.000066299754131,  0.000008480865679}, //(19,20)
    };

    // if t is "small", use quintic approximation
    if (t < 20){
        const NN::type T0 = std::round(t);
        const auto IDX = static_cast<unsigned>(std::floor(t));
        NN::type correction = quintics[IDX][0]     * (t - T0)
                            - quintics[IDX][1] / 2 * (pow(t,2) - pow(T0,2))
                            + quintics[IDX][2] / 3 * (pow(t,3) - pow(T0,3))
                            - quintics[IDX][3] / 4 * (pow(t,4) - pow(T0,4))
                            + quintics[IDX][4] / 5 * (pow(t,5) - pow(T0,5))
                            - quintics[IDX][5] / 6 * (pow(t,6) - pow(T0,6));
        return (numeric_integral[static_cast<unsigned>(T0)] + correction);
    }

    // if t is sufficiently large, use asymptotic approximation
    else{
        NN::type correction = 0.5 * log(2/M_PI) * (t-20)
                              - (pow(t,3) - pow(20,3))/6
                              - t * log(t + sqrt(t*t + 4)) + t * log(20 + sqrt(20*20 + 4))
                              + sqrt(t*t + 4) - sqrt(20*20 + 4);
        return (numeric_integral[20] + correction);
    }

}
