#ifndef DNAX_LIBRARY_ALGEBRAICOPERATIONS_H
#define DNAX_LIBRARY_ALGEBRAICOPERATIONS_H


#include "Matrix.h"
#include "VectorClass.h"

/******************************************* VECTOR SUM ***************************************************/

/// Sum of two vectors
VectorClass operator+(const VectorClass& vec1, const VectorClass& vec2);

/// Sum of scalar and vector
VectorClass operator+(NN::type scalar, const VectorClass& vec);

/// Sum of vector and scalar
VectorClass operator+(const VectorClass& vec, NN::type scalar);


/******************************************* VECTOR DIFFERENCE ********************************************/

/// Difference of two vectors
VectorClass operator-(const VectorClass& vec1, const VectorClass& vec2);

/// Difference of scalar and vector
VectorClass operator-(NN::type scalar, const VectorClass& vec);

/// Difference of vector and scalar
VectorClass operator-(const VectorClass& vec, NN::type scalar);


/******************************************* VECTOR PRODUCT ***********************************************/

/// Elementwise product of vector
VectorClass operator*(const VectorClass& vec1, const VectorClass& vec2);

/// Product of scalar and vector
VectorClass operator*(NN::type scalar, const VectorClass& vec);

/// Product of vector and scalar
VectorClass operator*(const VectorClass& vec, NN::type scalar);


/******************************************* VECTOR DIVISION **********************************************/

/// Elementwise division of vector
VectorClass operator/(const VectorClass& vec1, const VectorClass& vec2);

/// Division of vector by scalar
VectorClass operator/(const VectorClass& vec, NN::type scalar);


/******************************************* VECTOR OPERATIONS ********************************************/

/// Square power of vector
VectorClass pow2(const VectorClass& vec);

/// Square root of vector
VectorClass sqrt(const VectorClass& vec);

/// Exponential of vector
VectorClass exp(const VectorClass& vec);

/******************************************* MATRIX OPERATIONS ********************************************/

/// Matrix divided by a scalar
Matrix operator/(const Matrix& mat, NN::type scalar);

/******************************************* MATRIX MULTIPLICATION ****************************************/

/// Matrix-Matrix multiplication
Matrix operator*(const Matrix& mat1, const Matrix& mat2);

/// Matrix-Matrix multiplication with no-aliasing
/**
 * Multiplication operator that reveals to be faster for medium-large matrices.
 * Usually the result of the multiplication of two matrices is stored in a
 * temporary matrix and then assigned to the first member through the assignment
 * operator, which involves a further copy of the matrix itself. In this case the results
 *  are directly stored in the #where matrix.
 * @param mat1:     LHS of the multiplication
 * @param mat2:     RHS of the multiplication
 * @param where:    matrix where the result has to be stored
 */
void mult_in_place(const Matrix& mat1, const Matrix& mat2, Matrix& where);


/// Matrix-Vector multiplication
VectorClass operator*(const Matrix& mat, const VectorClass& vec);

/// Matrix-Vector multiplication with no-aliasing
/**
 * Multiplication operator that reveals to be faster for medium-large matrices.
 * Usually the result of the multiplication of matrix-vector is stored in a
 * temporary vector and then assigned to the first member through the assignment
 * operator, which involves a further copy of the vector itself. In this case the results
 * are directly stored in the #where matrix.
 * @param mat:      LHS of the multiplication
 * @param vec:      RHS of the multiplication
 * @param where:    vector where the result has to be stored
 */
void mult_in_place(const Matrix& mat, const VectorClass& vec, VectorClass& where);



/// Vector-Matrix multiplication with no-aliasing
/**
 * \note The returned vector should be interpreted as \a row \a vector
 */
VectorClass operator*(const VectorClass& vec, const Matrix& mat);

/// Vector-Matrix multiplication with no-aliasing
/**
 * Multiplication operator that reveals to be faster for medium-large matrices.
 * Usually the result of the multiplication of matrix-vector is stored in a
 * temporary vector and then assigned to the first member through the assignment
 * operator, which involves a further copy of the vector itself. In this case the results
 * are directly stored in the #where matrix.
 * \note The returned vector should be interpreted as \a row \a vector
 * @param vec:      LHS of the multiplication
 * @param mat:      RHS of the multiplication
 * @param where:    vector where the result has to be stored
 */
void mult_in_place(const VectorClass& vec, const Matrix& mat, VectorClass& where);

/// Optimized A += B*C
/**
 * Performs A += B*C without creating the rhs and assigning it
 */
void mult_add_in_place(const Matrix& mat1, const Matrix& mat2, Matrix& where);

/// Optimized product A * D, with D diagonal matrix
void mult_AD(const Matrix& mat, const Matrix& diag, Matrix& where);

/// Optimized product v * D, with v (row) vector and D diagonal matrix
void mult_vD(const VectorClass& v, const Matrix& diag, VectorClass& where);

#endif //DNAX_LIBRARY_ALGEBRAICOPERATIONS_H
