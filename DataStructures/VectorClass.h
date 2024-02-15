/// Vector in R^n
/**
 * Class that implements a naive numeric vector, whose components are allocated and managed
 * dynamically in the heap.
 */

#ifndef DNAX_LIBRARY_VECTORCLASS_H
#define DNAX_LIBRARY_VECTORCLASS_H

#include <vector>
#include "../global.h"


class VectorClass {

private:

    const bool DEBUG = true;

    /********************************************** ATTRIBUTES **************************************************/

protected:

    /// Pointer to the elements of the vector
    NN::type* components = nullptr;

    /// Number of elements in the array
    /**
     * It coincides with the length of the dynamic array pointed by #components
     */
    unsigned numel = 0;

    /********************************************** CONSTRUCTORS ************************************************/

public:

    /// Default void constructor
    VectorClass() = default;

    /// Constructor: initialization with initializer list and size
    /**
     * Constructs a vector from a given initializer_list and its size.
     * In case the length of the initializer_list is greater than size, the
     * exceeding elements are not considered. In case the length of the
     * initializer_list is lower than size, the missing elements are initialized to 0
     * @param length_:      required size for the vector
     * @param elems_:       initializer_list with the elements
     */
    VectorClass(unsigned length_, std::initializer_list<NN::type> elems_);


    /// Constructor: initialization with initializer list
    /**
     * Constructs a vector from a given initializer_list. Size is
     * deduced from the one of the initializer_list.
     * @param elems_:       initializer_list with the elements
     */
    explicit VectorClass(std::initializer_list<NN::type> elems_);


    /// Constructor: pure allocation
    /**
     * Allocate an array of given size in the heap without initializing it.
     * May be useful to avoid double-initialization.
     * @param numel_:  size of the array
     */
    explicit VectorClass(unsigned numel_);


    /// Constructor: initialization with a unique given value
    /**
     * Allocate an array of given size in the heap and initialize all the elements
     * with a given value (usually 0).
     * @param numel_:   size of the array
     * @param value_:   initializer value
     */
    VectorClass(unsigned numel_, NN::type value_);


    /// Constructor: random initialization
    /**
     * Allocate an array of given size in the heap and randomly initialize all the elements.
     * It is possible to choose the distribution for the random initialization.
     * @param numel_:       size of the vector
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
    VectorClass(unsigned numel_, char dist_name_, unsigned seed_,
             NN::type val1 = 0.0, NN::type val2 = 1.0);


    /// Copy-constructor
    /**
     * Instantiate a new #VectorClass by copying the provided one. The array #components is
     * copied and not shared, so the two objects are independent.
     * @param rhs: vector to be copied
     */
    VectorClass(const VectorClass& rhs);

    /********************************************** DESTRUCTORS *************************************************/

public:

    /// Destructor of the object
    /**
     * If #release is true, the #components array is released. Otherwise, the pointed array
     * is shared with another element (that was the "real creator" of that array) which should
     * already being using that array and which is in charge of destructing it.
     */
    ~VectorClass();


    /********************************************** OPERATORS ***************************************************/

public:

    /// Assignment operator
    /**
     * Assign the RHS to the LHS. The array #components is
     * copied and not shared, so the two objects are independent.
     * @param rhs: vector to be copied
     * @return  reference to the copied vector
     */
    VectorClass& operator=(const VectorClass& rhs);


    /// Access operator (return copy)
    /**
     * Access the i-th element of the #components array.
     * \note No check on the well-position of the index is performed
     * @param idx: index in the array
     * @return copy of the i-th element of the array
     */
    NN::type operator()(unsigned idx) const;


    /// Access operator (return copy)
    /**
     * Access the i-th element of the #components array.
     * @param idx: index in the array
     * @return copy of the i-th element of the array
     */
    NN::type at(unsigned idx) const;


    /// Access operator (return reference)
    /**
     * Access the i-th element of the #components array.
     * @param idx: index in the array
     * @return reference to the i-th element of the array
     */
    NN::type& operator()(unsigned idx);


    /// Access operator (return reference)
    /**
     * Access the i-th element of the #components array.
     * \note No check on the well-position of the index is performed
     * @param idx: index in the array
     * @return reference to the i-th element of the array
     */
    NN::type& at(unsigned idx);

    /// Return pointer to component array
    NN::type* data(void) const;


    /// In-place sum operator
    /**
     * In-place elementwise addition
     * @param rhs: vector to be added
     * @return reference to the obtained vector
     */
    VectorClass& operator+=(const VectorClass& rhs);

    /// In-place difference operator
    /**
     * In-place elementwise subtraction
     * @param rhs: vector to be subtracted
     * @return reference to the obtained vector
     */
    VectorClass& operator-=(const VectorClass& rhs);

    /// In-place product operator
    /**
     * In-place elementwise multiplication
     * @param scalar: multiplication coefficient
     * @return reference to the obtained vector
     */
    VectorClass& operator*=(NN::type scalar);

    /// In-place division operator
    /**
     * In-place elementwise division
     * @param scalar: division coefficient
     * @return reference to the obtained vector
     */
    VectorClass& operator/=(NN::type scalar);

    /********************************************** METHODS *****************************************************/

public:

    /// Print
    /**
     * Print the %elements of the vector
     */
    void print(bool as_row = false) const;

    /// Export and write on a csv/txt file
    /**
     * Print the %elements of the vector by using the provided out_stream. Exported as row vector.
     */
    void print(std::ofstream& out_stream) const;

    /// Return the length of the stored array
    unsigned length(void) const;

    /// Set all the elements to zero, preserving the size
    void reset(void) const;

};

#endif //DNAX_LIBRARY_VECTORCLASS_H
