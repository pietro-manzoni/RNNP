
#include "LinearModel.h"

#include <cmath>
#include <algorithm>

#include "../DataStructures/VectorClass.h"
#include "../DataStructures/Matrix.h"
#include "../DataStructures/AlgebraicOperations.h"

/********************************************** METHODS *****************************************************/

void LinearModel::params(void) const{

    std::cout << "Betas for Linear Model:" << std::endl;
    betas.print();
    std::cout << std::endl;

}

void LinearModel::fit(const std::vector<VectorClass>& regressors, const std::vector<NN::type>& regressand,
    bool remove_outliers){

    // creating regressors matrix
    Matrix regressors_mat(regressors.size(), regressors[0].length());
        for (unsigned i = 0; i < regressors_mat.rows(); ++i)
            for (unsigned j = 0; j < regressors_mat.cols(); ++j)
                regressors_mat(i,j) = regressors[i](j);

    // creating regressand vector
    VectorClass regressand_vec(regressand.size());
    for (unsigned i = 0; i < regressand_vec.length(); ++i)
        regressand_vec(i) = regressand[i];

    if (remove_outliers)
        outliers_removal(regressors_mat, regressand_vec);

    // creating normal equations (A^T A x = A^T b)
    Matrix regressors_mat_t = regressors_mat.transpose();
    Matrix A = regressors_mat_t * regressors_mat;
    VectorClass b = regressors_mat_t * regressand_vec;

    // finding betas
    betas = LUP_solve(A, b);

}

std::vector<NN::type> LinearModel::predict(const std::vector<VectorClass>& regressors){

    // creating regressors matrix
    Matrix regressors_mat(regressors.size(), regressors[0].length());
    for (unsigned i = 0; i < regressors_mat.rows(); ++i)
        for (unsigned j = 0; j < regressors_mat.cols(); ++j)
            regressors_mat(i,j) = regressors[i](j);

    // predicting
    VectorClass prediction_vec = regressors_mat * betas;

    // converting output value
    std::vector<NN::type> prediction(prediction_vec.length());
    for (unsigned i = 0; i < prediction_vec.length(); ++i)
        prediction[i] = prediction_vec(i);

    return prediction;

}


VectorClass LinearModel::LUP_solve(Matrix& A, VectorClass& b){

    //check
    if (A.rows() != b.length()){
        std::cerr << "Matrix and vector size are not compatible" << std::endl;
        exit(3);
    }

    const double TOL = 1e-10; //Tolerance for considering a matrix as SINGULAR:
    // if the greatest available pivot (that can be chosen with a row permutation)
    // is less than TOL in absolute value, matrix is declared singular and it is not
    // decomposed

    const unsigned N = A.rows(); //cache value
    VectorClass P(N,0);  //initialize permutation vector
    VectorClass x(N,0); //initialize solution vector


    // LUP decomposition

    for (unsigned i = 0; i < N; i++)
        P(i) = i; //initialize P sequentially, it will represents the actual order in which rows are processed

    for (unsigned i = 0; i < N; i++) {
        NN::type maxA = 0, absA = 0;
        unsigned imax = i; // ideally, I would work on the (i,i) pivot (but maybe a permutation will take place)

        // having fixed the i-th column, I choose which is the k-th row (k>=i) that should be processed in order
        // to have the greatest possible pivot (in absolute value)
        for (unsigned k = i; k < N; k++)
            if ( (absA = std::fabs(A(k,i)) ) > maxA) {
                maxA = absA;    // maximum value of A(k,i) for k varying in [i,N)
                imax = k;       // row for which the maximum is attained
            }

        if (maxA < TOL) {
            std::cerr << "Matrix is too close to be singular" << std::endl;
            exit(3); // The maximum available pivot is too small. Exit.
        }

        // if the choice of the pivot requires a permutation, I keep track of this
        if (imax != i)
            std::swap( P(imax), P(i) );

        // use Gauss elimination method
        for (unsigned j = i + 1; j < N; j++) {
            A( P(j), i) /= A( P(i), i);
            for (unsigned k = i + 1; k < N; k++)
                A( P(j), k) -= A( P(j), i) * A( P(i), k);
        }

    }

    // At this point the matrix A is dense and contains:
    // - the U matrix in its upper triangular part (including the diagonal)
    // - the L matrix in its lower triangular part (excluding the diagonal, which is unary by definition)
    // P contains the permutation order of the rows

    // FORWARD substitution
    for (unsigned i = 0; i < N; ++i) {
        x(i) = b(P(i));
        for (unsigned k = 0; k < i; ++k)
            x(i) -= A(P(i),k) * x(k);
    }

    // BACKWARD substitution
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) { //because i>=0, 'unsigned' type cannot be used
        for (unsigned k = i + 1; k < N; ++k)
            x(i) -= A(P(i),k) * x(k);
        x(i) = x(i) / A(P(i),i);
    }

    return x;

}


void LinearModel::outliers_removal(Matrix& A, VectorClass& b) const{
// to avoid leverage effect in linear model, we want to remove the rows
// corresponding to outliers in the training regressand.
// It is easy to show that, using OLS, this is equivalent to putting to zero
// the selected rows in the design matrix.
// Outliers are selected by means of INTERQUANTILE RANGE

    // create a copy of b (a std::vector)
    std::vector<NN::type> b_vec(b.length());
    for (unsigned i = 0; i < b.length(); ++i)
        b_vec[i] = b(i);

    //sort the copy
    std::sort(b_vec.begin(), b_vec.end());

    // find 1st and 3rd quartiles
    const NN::type Q25 = b_vec[b_vec.size()*0.25];
    const NN::type Q75 = b_vec[b_vec.size()*0.75];
    const NN::type IQR = Q75 - Q25;


    // find outliers and set corresponding row of design matrix to 0
    for (unsigned i = 0; i < b.length(); ++i){
        if ((b(i) > Q75 + 3*IQR) || (b(i) < Q25 - 3*IQR)){
            for (unsigned j = 0; j < A.cols(); ++j){
                A(i,j) = 0;
            }
        }
    }

    return;

}
