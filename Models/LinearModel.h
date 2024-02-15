#ifndef DNAX_LIBRARY_LINEARMODEL_H
#define DNAX_LIBRARY_LINEARMODEL_H


#include <vector>

#include "../global.h"
#include "../DataStructures/VectorClass.h"
#include "../DataStructures/Matrix.h"


class LinearModel {

    /********************************************** ATTRIBUTES **************************************************/

private:

    VectorClass betas;

    /****************************************** CONSTRUCTORS ********************************************/

public:

    /// Default void constructor
    LinearModel() = default;

    /********************************************** METHODS *****************************************************/

private:

  /// Solve a linar system using LUP factorization
  /**
   * Solve a linear system employing the LU (Lower-Upper triangular) factorization
   * of the matrix with pivotal Permutation.
   * @param A:      design matrix of linear model
   * @param b:      regressand variable
   */
    VectorClass LUP_solve(Matrix& A, VectorClass& b);

    /// Adapt matrix A so that outliers are not considered in linear regression (OLS)
    /**
     * First, vector b is analysed. The ouliers (identified according to IQR criterion)
     * are spotted and their indexes stored. Then each row of the design matrix
     * that corresponds to an outlier is set to 0. This can be proved to be
     * mathematically equivalent (in terms of OLS projection) to the removal of these rows.
     *
     * @param A:      design matrix of linear model
     * @param b:      regressand variable
     */
    void outliers_removal(Matrix& A, VectorClass& b) const;

public:

    /// Print betas of the regression
    void params(void) const;

    /// Fit linear model
    void fit(const std::vector<VectorClass>& regressors, const std::vector<NN::type>& regressand, bool remove_outliers = true);

    /// Predict with linear model
    std::vector<NN::type> predict(const std::vector<VectorClass>& regressors);

};


#endif //DNAX_LIBRARY_LINEARMODEL_H
