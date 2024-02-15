#include <iomanip>
#include <random>

#include "Matrix.h"

/********************************************** CONSTRUCTORS ************************************************/

//initialization with initializer list and size
Matrix::Matrix(unsigned rows_, unsigned cols_, std::initializer_list<NN::type> elems_) :
        components ( new NN::type[rows_ * cols_]() ), numel(rows_ * cols_),
        n_rows(rows_), n_cols(cols_) //components is initialized with 0's
{
    // select the mininum between numel (total number of elements in matrix) and length of list
    const unsigned IDX = std::min( static_cast<unsigned>(elems_.size()), numel );

    // copy elements into matrix
    auto iter = elems_.begin();
    for (unsigned i = 0; i < IDX; ++i )
        components[i] = *(iter++);
}


//pure allocation and initialization to zero
Matrix::Matrix(unsigned rows_, unsigned cols_) :
        components (new NN::type[rows_ * cols_]()), numel(rows_ * cols_),
        n_rows(rows_), n_cols(cols_)
{
    std::fill_n(components, numel, 0); // initialize to zero
}


//initialization with a unique given value
Matrix::Matrix(unsigned rows_, unsigned cols_, NN::type value_) :
        Matrix(rows_, cols_)
{
    std::fill_n(components, numel, value_);
}


//random initialization
Matrix::Matrix(unsigned rows_, unsigned cols_, char dist_name_, unsigned seed_, NN::type val1,
               NN::type val2) :
        Matrix(rows_, cols_)
{
    switch (dist_name_) {
        //uniform distribution
        case 'u' : {
            std::mt19937 generator(seed_);
            std::uniform_real_distribution<NN::type> distribution(val1, val2);
            for (std::size_t i = 0; i < numel; ++i)
                components[i] = distribution(generator);
            break;
        }

        //normal distribution
        case 'n' : {
            std::mt19937 generator(seed_);
            std::normal_distribution<NN::type> distribution(val1, val2);
            for (std::size_t i = 0; i < numel; ++i)
                components[i] = distribution(generator);
            break;
        }

        default: {
            std::cerr << "Chosen distribution symbol is not valid." << std::endl;
            exit(99);
        }

    }
}


// Copy-constructor
Matrix::Matrix(const Matrix& rhs) : Matrix(rhs.n_rows, rhs.n_cols) // create suitable matrix panned with 0's
{
    for (unsigned i = 0; i < numel; ++i)
        components[i] = rhs.components[i];
}


/********************************************** DESTRUCTORS *************************************************/

Matrix::~Matrix(){
    delete[] components; // avoid memory leakage!
}


/********************************************** OPERATORS ***************************************************/

// assignment operator
Matrix& Matrix::operator=(const Matrix& rhs){
    if (this != &rhs) { //no need for assigning an element to itself...

        // if necessary, reallocate element with right size
        if (numel != rhs.numel){
            //std::cout << "DELETED: " << components <<  std::endl;
            delete[] components;
            components = new NN::type[rhs.numel];
            //std::cout << "CREATED: " << components <<  std::endl;
            numel = rhs.numel;
        }

        // updating size
        n_rows = rhs.n_rows;
        n_cols = rhs.n_cols;

        // copying contents
        for (unsigned i = 0; i < numel; ++i)
            components[i] = rhs.components[i];
    }

    return *this;
}


NN::type Matrix::operator()(unsigned i, unsigned j) const{
    if (i >= n_rows || j >= n_cols){
        std::cerr << "Error in matrix operator () : idx out of bounds" << std::endl;
        exit(2);
    }
    return components[i * n_cols + j];
}


NN::type Matrix::at(unsigned i, unsigned j) const{
    if (i >= n_rows || j >= n_cols){
        std::cerr << "Error in matrix operator at : idx out of bounds" << std::endl;
        exit(2);
    }
    return components[i * n_cols + j];
}


NN::type& Matrix::operator()(unsigned i, unsigned j){
    if (i >= n_rows || j >= n_cols){
        std::cerr << "Error in matrix operator () [REFERENCE VERSION] : idx out of bounds" << std::endl;
        exit(2);
    }
    return components[i * n_cols + j];
}


NN::type& Matrix::at(unsigned i, unsigned j){
    if (i >= n_rows || j >= n_cols){
        std::cerr << "Error in matrix operator at [REFERENCE VERSION] : idx out of bounds" << std::endl;
        exit(2);
    }
    return components[i * n_cols + j];
}


Matrix& Matrix::operator+=(const Matrix& rhs){
    if (rhs.n_rows != n_rows || rhs.n_cols != n_cols){
        std::cerr << "Error in matrix operator += : incompatible sizes" << std::endl;
        exit(2);
    }
    for (unsigned i = 0; i < numel; ++i)
        components[i] += rhs.components[i];
    return (*this);
}


Matrix& Matrix::operator-=(const Matrix& rhs){
    if (rhs.n_rows != n_rows || rhs.n_cols != n_cols){
        std::cerr << "Error in matrix operator -= : incompatible sizes" << std::endl;
        exit(2);
    }
    for (unsigned i = 0; i < numel; ++i)
        components[i] -= rhs.components[i];
    return (*this);
}


Matrix& Matrix::operator*=(NN::type scalar){
    for (unsigned i = 0; i < numel; ++i)
        components[i] *= scalar;
    return (*this);
}


Matrix& Matrix::operator/=(NN::type scalar){
    for (unsigned i = 0; i < numel; ++i)
        components[i] /= scalar;
    return (*this);
}


/********************************************** METHODS *****************************************************/


void Matrix::print() const{
    for (unsigned i = 0; i < n_rows; ++i) {
        for (unsigned j = 0; j < n_cols; ++j)
            std::cout << std::right << std::setw(NN::SPACING) << components[i*n_cols + j] ;
        std::cout << std::endl;
    }
    std::cout << std::endl;

}


void Matrix::print(std::ofstream& out_stream) const{
    out_stream << n_rows << ',' << n_cols << std::endl;
    for (unsigned i = 0; i < n_rows; ++i) {
        for (unsigned j = 0; j < n_cols; ++j)
            out_stream << components[i*n_cols + j] << ',';
        out_stream << std::endl;
    }
    out_stream << std::endl;
}


unsigned Matrix::rows() const{
    return n_rows;
}


unsigned Matrix::cols() const{
    return n_cols;
}


void Matrix::reset() const{
    for (unsigned i = 0; i < numel; ++i)
        components[i] = 0;
}


Matrix Matrix::transpose() const{
    Matrix trans_matrix(n_cols, n_rows);
    for (unsigned i = 0; i < n_cols; ++i)
        for (unsigned j = 0; j < n_rows; ++j)
            trans_matrix(i,j) = at(j,i);

    return trans_matrix;
}
