/// Template class for Circular Buffer
/**
 * Class that implements a Circular Buffer, a data structure that can be useful to store
 * a fixed number of elements and replace last one with a "rolling" procedure.
 * Basically, a vector with smart indexing.
 */

#ifndef DNAX_LIBRARY_CIRCULARBUFFER_H
#define DNAX_LIBRARY_CIRCULARBUFFER_H


#include <vector>
#include <iostream>

template <typename T>
class CircularBuffer{

    /********************************************** ATTRIBUTES **************************************************/

private:

    /// Container (std::vector) that serves as buffere
    std::vector<T> buff;

    /// Index to the current position for insertion
    unsigned current_idx;

public:

    /// Size of the circular buffer
    /**
     * Defined during the creation of the object.
     */
    const unsigned N;


    /********************************************** CONSTRUCTOR *************************************************/

public:

    /// Constructor
    /**
     * Only viable construction of the object is by declaring its size, that CANNOT be modified later.
     * @param N_ size of the circular buffer
     */
    explicit CircularBuffer(unsigned N_) : N(N_), buff(N_), current_idx(0) {};


    /// Deleting copy-constructor
    CircularBuffer(const CircularBuffer&) = delete;

    /********************************************** OPERATORS *************************************************/

public:

    /// Assignment operator
    /**
     * Assign the RHS to the LHS. The contained objects of type <T> are
     * copied and not shared, so the two circular buffers are independent.
     * @param cb
     * @return reference to the copied buffer
     */
    CircularBuffer& operator=(const CircularBuffer& cb){

        // size of circular buffer is defined during construction and cannot be modified
        if (cb.N != N) {
            std::cerr << "Impossible to assign Circular Buffers w/ different size" << std::endl;
            exit(1);
        }

        // assigning elements
        if (this != &cb){
            for (unsigned i = N; i > 0; --i)
                insert( cb(i) );
        }

        return *this;
    }


    /// Read a given element in the buffer
    /**
     * Retrieve a const reference to the object in position #idx in the buffer.
     * \note The output of the function is a CONSTANT REFERENCE: in other words, we have a READ-ONLY feature.
     * @param idx: time-lag. #idx equal 1 means "the object inserted at time t-1", i.e. the most recent insertion.
     * @return const reference to the
     */
    const T& operator()(unsigned idx) const{ //equivalent to "read_element"
        if (idx == 0)
            std::cerr << "Accessing to index 0 in CB: probable error" << std::endl;
        if (N==0){
            std::cerr << "Unable to access. CB has size 0." << std::endl;
            exit(2);
        }
        return buff[ (N + current_idx - idx) % N ];
    }


    /************************************************ METHODS ***************************************************/

    /// Read a given element in the buffer
    /**
     * Retrieve a const reference to the object in position #idx in the buffer.
     * \note The output of the function is a CONSTANT REFERENCE: in other words, we have a READ-ONLY feature.
     * @param idx: time-lag. #idx equal 1 means "the object inserted at time t-1", i.e. the most recent insertion.
     * @return const reference to the
     */
    const T& read_element(unsigned idx) const{
        if (idx == 0)
            std::cerr << "Accessing to index 0 in CB: probable error" << std::endl;
        if (N==0){
            std::cerr << "Unable to access. CB has size 0." << std::endl;
            exit(2);
        }
        return buff[ (N + current_idx - idx) % N ];
    }


    /// Inserting a new element in the circular buffer.
    /**
     * Inserting a new element.
     * @param new_elem: element to be inserted.
     */
    void insert(const T& new_elem) {
        if (N==0) {
            std::cerr << "Unable to access " << std::endl;
            exit(2);
        }
        buff[ current_idx ] = new_elem;
        current_idx = (current_idx + 1) % N;
    }

    /// Print
    /**
     * Print the entire content of the circular buffer.
     */
    void print(void) const{
        for (unsigned i = 1; i <= N; ++i) {
            std::cout << i << std::endl;
            read_element(i).print();
        }
    }

    /// Reset buffer
    /**
     *  Resetting all the elements in buffer. The <T> type must therefore admit a reset method, f.i. that sets all
     *  the components of <T> to zero or to a default value.
     */
    void reset(void){
        for (auto& elem : buff) {
            elem.reset();
        }
    }


};


#endif //DNAX_LIBRARY_CIRCULARBUFFER_H
