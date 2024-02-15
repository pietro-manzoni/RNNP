#include <cmath>

#include "AlgebraicOperations.h"

/******************************************* VECTOR SUM ***************************************************/

VectorClass operator+(const VectorClass& vec1, const VectorClass& vec2){
    if (vec1.length() != vec2.length()){
        std::cerr << "Wrong size for vector summation" << std::endl;
        exit(3);
    }

    VectorClass out_vec(vec1.length());
    for (unsigned i = 0; i < vec1.length(); ++i)
        out_vec(i) = vec1(i) + vec2(i);
    return out_vec;
}

VectorClass operator+(NN::type scalar, const VectorClass& vec){

    VectorClass out_vec(vec.length());
    for (unsigned i = 0; i < vec.length(); ++i)
        out_vec(i) = scalar + vec(i);
    return  out_vec;
}

VectorClass operator+(const VectorClass& vec, NN::type scalar){
    return  (scalar + vec); // commutativity
}

/******************************************* VECTOR DIFFERENCE ********************************************/

VectorClass operator-(const VectorClass& vec1, const VectorClass& vec2){
    if (vec1.length() != vec2.length()){
        std::cerr << "Wrong size for vector difference" << std::endl;
        exit(3);
    }

    VectorClass out_vec(vec1.length());
    for (unsigned i = 0; i < vec1.length(); ++i)
        out_vec(i) = vec1(i) - vec2(i);
    return out_vec;
}

VectorClass operator-(NN::type scalar, const VectorClass& vec){

    VectorClass out_vec(vec.length());
    for (unsigned i = 0; i < vec.length(); ++i)
        out_vec(i) = scalar - vec(i);
    return  out_vec;
}

VectorClass operator-(const VectorClass& vec, NN::type scalar){

    VectorClass out_vec(vec.length());
    for (unsigned i = 0; i < vec.length(); ++i)
        out_vec(i) = vec(i) - scalar;
    return  out_vec;
}


/******************************************* VECTOR PRODUCT ***********************************************/

VectorClass operator*(const VectorClass& vec1, const VectorClass& vec2){
    if (vec1.length() != vec2.length()){
        std::cerr << "Wrong size for vector product" << std::endl;
        exit(3);
    }

    VectorClass out_vec(vec1.length());
    for (unsigned i = 0; i < vec1.length(); ++i)
        out_vec(i) = vec1(i) * vec2(i);
    return out_vec;
}

VectorClass operator*(NN::type scalar, const VectorClass& vec){

    VectorClass out_vec(vec.length());
    for (unsigned i = 0; i < vec.length(); ++i)
        out_vec(i) = scalar * vec(i);
    return  out_vec;
}

VectorClass operator*(const VectorClass& vec, NN::type scalar){
    return (scalar*vec); //commutativity
}

/******************************************* VECTOR DIVISION **********************************************/

VectorClass operator/(const VectorClass& vec1, const VectorClass& vec2){
    if (vec1.length() != vec2.length()){
        std::cerr << "Wrong size for vector division" << std::endl;
        exit(3);
    }

    VectorClass out_vec(vec1.length());
    for (unsigned i = 0; i < vec1.length(); ++i )
        out_vec(i) = vec1(i) / vec2(i);
    return out_vec;
}

VectorClass operator/(const VectorClass& vec, NN::type scalar){

    VectorClass out_vec(vec.length());
    for (unsigned i = 0; i < vec.length(); ++i )
        out_vec(i) = vec(i) / scalar;
    return out_vec;
}


/******************************************* VECTOR OPERATIONS ********************************************/

VectorClass pow2(const VectorClass& vec){

    VectorClass out_vec(vec.length());
    for (unsigned i = 0; i < vec.length(); ++i)
        out_vec(i) = vec(i) * vec(i);
    return  out_vec;
}

VectorClass sqrt(const VectorClass& vec){

    VectorClass out_vec(vec.length());
    for (unsigned i = 0; i < vec.length(); ++i)
        out_vec(i) = std::sqrt( vec(i) );
    return  out_vec;
}

VectorClass exp(const VectorClass& vec) {

    VectorClass out_vec(vec.length());
    for (unsigned i = 0; i < vec.length(); ++i)
        out_vec(i) = std::exp(vec(i));
    return out_vec;
}

/******************************************* MATRIX OPERATIONS ********************************************/

/// Matrix divided by a scalar
Matrix operator/(const Matrix& mat, NN::type scalar){

    Matrix out_mat(mat.rows(), mat.cols());
    for (unsigned i = 0; i < mat.rows(); ++i)
        for (unsigned j = 0; j < mat.cols(); ++j)
            out_mat(i,j) = mat(i,j)/scalar;
    return out_mat;
}


/******************************************* MATRIX MULTIPLICATION ****************************************/

Matrix operator*(const Matrix& mat1, const Matrix& mat2){

    if (mat1.cols() != mat2.rows()){
        std::cerr << "Wrong size for Matrix-Matrix multiplication" << std::endl;
        exit(3);
    }

    Matrix out_mtrx(mat1.rows(), mat2.cols());

    for (unsigned i = 0; i < mat1.rows(); ++i)
        for (unsigned j = 0; j < mat2.cols(); ++j) {
            out_mtrx(i, j) = 0;
            for (unsigned k = 0; k < mat1.cols(); ++k)
                out_mtrx(i,j) += mat1(i,k) * mat2(k,j);
        }

    return  out_mtrx;
}


void mult_in_place(const Matrix& mat1, const Matrix& mat2, Matrix& where){

    if (mat1.cols() != mat2.rows()){
        std::cerr << "Wrong size for Matrix-Matrix multiplication" << std::endl;
        exit(3);
    }
    if (mat1.rows() != where.rows() || mat2.cols() != where.cols()){
        std::cerr << "Wrong size for WHERE matrix" << std::endl;
        exit(3);
    }

    for (unsigned i = 0; i < mat1.rows(); ++i)
        for (unsigned j = 0; j < mat2.cols(); ++j) {
            where(i, j) = 0;
            for (unsigned k = 0; k < mat1.cols(); ++k)
                where(i,j) += mat1(i,k) * mat2(k,j);
        }

}

VectorClass operator*(const Matrix& mat, const VectorClass& vec){

    if (mat.cols() != vec.length()){
        std::cerr << "Wrong size for Matrix-Vector multiplication" << std::endl;
        exit(3);
    }

    VectorClass out_vec(mat.rows());
    for (unsigned i = 0; i < mat.rows(); ++i){
        out_vec(i) = 0;
        for (unsigned j = 0; j < mat.cols(); ++j)
            out_vec(i) += mat(i,j) * vec(j);
    }
    return  out_vec;
}

void mult_in_place(const Matrix& mat, const VectorClass& vec, VectorClass& where){

    if (mat.cols() != vec.length()){
        std::cerr << "Wrong size for Matrix-Vector multiplication" << std::endl;
        exit(3);
    }
    if (mat.rows() != where.length()){
        std::cerr << "Wrong size for WHERE Vector" << std::endl;
        exit(3);
    }

    for (unsigned i = 0; i < mat.rows(); ++i){
        where(i) = 0;
        for (unsigned j = 0; j < mat.cols(); ++j)
            where(i) += mat(i,j) * vec(j);
    }

}


VectorClass operator*(const VectorClass& vec, const Matrix& mat){

    if (vec.length() != mat.rows()){
        std::cerr << "Wrong size for Vector-Matrix multiplication" << std::endl;
        exit(3);
    }

    VectorClass out_vec(mat.cols());
    for (unsigned j = 0; j < mat.cols(); ++j){
        out_vec(j) = 0;
        for (unsigned i = 0; i < mat.rows(); ++i)
            out_vec(j) += vec(i) * mat(i,j);
    }
    return  out_vec;
}


void mult_in_place(const VectorClass& vec, const Matrix& mat, VectorClass& where){

    if (vec.length() != mat.rows()){
        std::cerr << "Wrong size for Vector-Matrix multiplication" << std::endl;
        exit(1);
    }
    if (mat.cols() != where.length()){
        std::cerr << "Wrong size for WHERE Vector" << std::endl;
        exit(3);
    }

    for (unsigned j = 0; j < mat.cols(); ++j){
        where(j) = 0;
        for (unsigned i = 0; i < mat.rows(); ++i)
            where(j) += vec(i) * mat(i,j);
    }
}


void mult_add_in_place(const Matrix& mat1, const Matrix& mat2, Matrix& where){

    if (mat1.cols() != mat2.rows()){
        std::cerr << "Wrong size for Matrix-Matrix multiplication" << std::endl;
        exit(3);
    }
    if (mat1.rows() != where.rows() || mat2.cols() != where.cols()){
        std::cerr << "Wrong size for WHERE matrix" << std::endl;
        exit(3);
    }

    for (unsigned i = 0; i < mat1.rows(); ++i)
        for (unsigned j = 0; j < mat2.cols(); ++j)
            for (unsigned k = 0; k < mat1.cols(); ++k)
                where(i,j) += mat1(i,k) * mat2(k,j);

}


void mult_AD(const Matrix& mat, const Matrix& diag, Matrix& where){

    if (mat.cols() != diag.rows()){
        std::cerr << "Wrong size for Matrix-Matrix multiplication" << std::endl;
        exit(3);
    }
    if (mat.rows() != where.rows() || diag.cols() != where.cols()){
        std::cerr << "Wrong size for WHERE matrix" << std::endl;
        exit(3);
    }

    for (unsigned i = 0; i < mat.rows(); ++i)
        for (unsigned j = 0; j < diag.cols(); ++j)
            where(i,j) = mat(i,j) * diag(j,j);
}


/// Optimized product v * D, with v (row) vector and D diagonal matrix
void mult_vD(const VectorClass& v, const Matrix& diag, VectorClass& where){

    if (v.length() != diag.rows()){
        std::cerr << "Wrong size for Vector-Matrix multiplication" << std::endl;
        exit(3);
    }
    if (v.length() != where.length()){
        std::cerr << "Wrong size for WHERE matrix" << std::endl;
        exit(3);
    }

    for (unsigned i = 0; i < v.length(); ++i)
        where(i) = v(i) * diag(i,i);

}