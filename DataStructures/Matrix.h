#ifndef DNAX_LIBRARY_MATRIX_H
#define DNAX_LIBRARY_MATRIX_H


#include "../global.h"

#include <iostream>
#include <fstream>
#include <initializer_list>


/// Two dimensional matrix
/**
 * Class that implements a naive numeric matrix, whose components are allocated and managed
 * dynamically as a unique array (ordered by rows) in the heap.
 */

class Matrix {

    /********************************************** ATTRIBUTES **************************************************/

protected:

    /// Pointer to the elements of the dynamic array
    NN::type* components = nullptr;

    /// Number of rows of the matrix
    unsigned n_rows = 0;

    /// Number of columns of the matrix
    unsigned n_cols = 0;

    /// Number of elements in the array
    /**
     * It coincides with the length of the dynamic array pointed by #components, and is equal
     * to #n_rows * #n_cols. Stored for convenience.
     */
    unsigned numel = 0;


    /****************************************** CONSTRUCTORS ********************************************/

public:

    /// Default void constructor
    Matrix() = default;

    /// Constructor: initialization with initializer list and row/column size
    /**
     * Constructs a matrix from a given initializer_list and its size.
     * In case the length of the initializer_list is greater than size, the
     * exceeding elements are not considered. In case the length of the
     * initializer_list is lower than size, the missing elements are initialized to 0
     * @param rows_:        required row size
     * @param cols_:        required column size
     * @param elems_:       initializer_list with the elements
     */
    Matrix(unsigned rows_, unsigned cols_, std::initializer_list<NN::type> elems_);

    /// Constructor: pure allocation
    /**
     * Allocate a matrix of given size in the heap without initializing it.
     * May be useful to avoid double-initialization.
     * @param rows_:        required row size
     * @param cols_:        required column size
     */
    Matrix(unsigned rows_, unsigned cols_);

    /// Constructor: initialization with a unique given value
    /**
     * Allocate an array of given size in the heap and initialize all the elements
     * with a given value (usually 0).
     * @param rows_:        required row size
     * @param cols_:        required column size
     * @param value_:       initializer value
     */
    Matrix(unsigned rows_, unsigned cols_, NN::type value_);

    /// Constructor: random initialization
    /**
     * Allocate an array of given size in the heap and randomly initialize all the elements.
     * It is possible to choose the distribution for the random initialization.
     * @param rows_:        required row size
     * @param cols_:        required column size
     * @param dist_name_:   char that indicates the distribution. Can be
     *                      - 'n' for normal
     *                      - 'u' for uniform
     * @param seed_:        seed for the generation of the distribution
     * @param val1:         feature of the distribution. In case of
     *                      - normal distribution, it represents the mean
     *                      - uniform distribution, it represents the lower bound
     * @param val2:         feature of the distribution. In case of
     *                      - normal distribution, it represents the standard deviation
     *                      - uniform distribution, it represents the upper bound
     */
    Matrix(unsigned rows_, unsigned cols_, char dist_name_, unsigned seed_,
           NN::type val1 = 0.0, NN::type val2 = 1.0);

    /// Copy-constructor
    /**
     * Instantiate a new #Matrix by copying the provided one. The array #components is
     * copied and not shared, so the two objects are independent.
     * @param rhs: matrix to be copied
     */
    Matrix(const Matrix& rhs);

    /******************************************* DESTRUCTORS **********************************************/

public:

    /// Destructor of the object
    /**
     * Deallocate the dynamic array, in order to avoid memory leakage.
     */
    ~Matrix();

    /****************************************** OPERATORS ************************************************/

public:

    /// Assignment operator
    /**
     * Assign the RHS to the LHS. The array #components is
     * copied and not shared, so the two objects are independent.
     * @param rhs: matrix to be copied
     * @return  reference to the copied matrix
     */
    Matrix& operator=(const Matrix& rhs);

    /// Access operator
    /**
     * Access the (i,j) element of the matrix
     * @param i: row-index in the array
     * @param j: column-index in the array
     * @return copy of the (i,j) element of the matrix
     */
    NN::type operator()(unsigned i, unsigned j) const;

    /// Access operator
    /**
     * Access the (i,j) element of the matrix
     * @param i: row-index in the array
     * @param j: column-index in the array
     * @return copy of the (i,j) element of the matrix
     */
    NN::type at(unsigned i, unsigned j) const;

    /// Access operator
    /**
     * Access the (i,j) element of the matrix
     * @param i: row-index in the array
     * @param j: column-index in the array
     * @return reference to the (i,j) element of the matrix
     */
    NN::type& operator()(unsigned i, unsigned j);

    /// Access operator
    /**
     * Access the (i,j) element of the matrix
     * @param i: row-index in the array
     * @param j: column-index in the array
     * @return reference to the (i,j) element of the matrix
     */
    NN::type& at(unsigned i, unsigned j);

    /// In-place sum operator
    /**
     * In-place elementwise addition
     * @param rhs: matrix to be added
     * @return reference to the obtained matrix
     */
    Matrix& operator+=(const Matrix& rhs);

    /// In-place difference operator
    /**
     * In-place elementwise subtraction
     * @param rhs: matrix to be subtracted
     * @return reference to the obtained matrix
     */
    Matrix& operator-=(const Matrix& rhs);

    /// In-place product operator
    /**
     * In-place elementwise multiplication
     * @param scalar: multiplication coefficient
     * @return reference to the obtained matrix
     */
    Matrix& operator*=(NN::type scalar);

    /// In-place division operator
    /**
     * In-place elementwise division
     * @param scalar: division coefficient
     * @return reference to the obtained matrix
     */
    Matrix& operator/=(NN::type scalar);


    /********************************************** METHODS *****************************************************/

public:

    /// Print
    /**
     * Print the elements of the matrix
     */
    void print(void) const;

    /// Export and write on a csv/txt file
    /**
     * Print the elements of the matrix by using the provided out_stream
     */
    void print(std::ofstream& out_stream) const;

    /// Transpose
    /**
     * @return transposed matrix
     */
    Matrix transpose(void) const;

    /// Return the number of rows of the stored matrix
    unsigned rows(void) const;

    /// Return the number of columns of the stored matrix
    unsigned cols(void) const;

    /// Set all the elements to zero, preserving the size
    void reset(void) const;

    /// Print Memory Address (DEBUG)
    void printmemoryaddress(void) const{ std::cout << components << std::endl; }

};


#endif //DNAX_LIBRARY_MATRIX_H
