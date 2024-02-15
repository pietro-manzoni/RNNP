#include <random>
#include <iomanip>
#include <fstream>

#include "VectorClass.h"


// initialization with initializer list and size
VectorClass::VectorClass(unsigned length_, std::initializer_list<NN::type> elems_) :
    components( new NN::type[length_]() ), numel(length_)  //components is initialized to 0
    {
        // select minimum between length of list and provided length
        const unsigned IDX = std::min( static_cast<unsigned>(elems_.size()), length_ );

        // copy elements in the vector
        auto iter = elems_.begin();
        for (unsigned i = 0; i < IDX; ++i )
            components[i] = *(iter++);

        // note: if provided elements are less than length_, the vector is panned with zeros
        //       if provided elements are more than length_, the excess ones are ignored
    }


// initialization with initializer list (and no size specified)
VectorClass::VectorClass(std::initializer_list<NN::type> elems_) :
        VectorClass(static_cast<unsigned>(elems_.size()), elems_) {}


// pure allocation (with zero initialization)
VectorClass::VectorClass(unsigned numel_) :
        components(new NN::type[numel_]), numel(numel_)
{
    std::fill_n(components, numel_, 0);
}


// initialization with a unique given value
VectorClass::VectorClass(unsigned numel_, NN::type value_) :
        VectorClass(numel_)
{
    std::fill_n(components, numel_, value_);
}


// random initialization
VectorClass::VectorClass(unsigned numel_, char dist_name_, unsigned seed_, NN::type val1, NN::type val2) :
        VectorClass(numel_)
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


// copy-constructor
VectorClass::VectorClass(const VectorClass& rhs) : VectorClass(rhs.numel)
{   // instantiate and overwrite elements
    for (unsigned i = 0; i < numel; ++i)
        components[i] = rhs.components[i];
}


/********************************************** DESTRUCTORS *************************************************/

VectorClass::~VectorClass(){
    delete[] components; // avoid memory leakage!
}


/********************************************** OPERATORS ***************************************************/

VectorClass& VectorClass::operator=(const VectorClass& rhs){
    if (this != &rhs) { //no need for assigning an element to itself...

        // if necessary, reallocate element with right size
        if (numel != rhs.numel){
            delete[] components;
            components = new NN::type[rhs.numel];
            numel = rhs.numel;
        }

        // copying content
        for (unsigned i = 0; i < numel; ++i)
            components[i] = rhs.components[i];
    }

    return *this;
}


NN::type VectorClass::operator()(unsigned idx) const{
    if (idx >= numel) {
        std::cerr << "Error in operator () : idx out of bounds" << std::endl;
        exit(1);
    }
    return components[idx];
}


NN::type VectorClass::at(unsigned idx) const{
    if (idx >= numel) {
        std::cerr << "Error in operator at : idx out of bounds" << std::endl;
        exit(1);
    }
    return components[idx];
}


NN::type& VectorClass::operator()(unsigned idx){
    if (idx >= numel) {
        std::cerr << "Error in operator () [REFERENCE VERSION] : idx out of bounds" << std::endl;
        exit(1);
    }
    return components[idx];
}


NN::type& VectorClass::at(unsigned idx){
    if (idx >= numel) {
        std::cerr << "Error in operator at [REFERENCE VERSION] : idx out of bounds" << std::endl;
        exit(1);
    }
    return components[idx];
}

NN::type* VectorClass::data(void) const{
    return  components;
}

VectorClass& VectorClass::operator+=(const VectorClass& rhs){
    if (numel != rhs.numel){
        std::cerr << "Error with operator += : Incompatible size" << std::endl;
        exit(1);
    }
    for (unsigned i = 0; i < numel; ++i)
        components[i] += rhs.components[i];
    return *this;
}


VectorClass& VectorClass::operator-=(const VectorClass& rhs){
    if (numel != rhs.numel){
        std::cerr << "Error with operator -= : Incompatible size" << std::endl;
        exit(1);
    }
    for (unsigned i = 0; i < numel; ++i)
        components[i] -= rhs.components[i];
    return *this;
}


VectorClass& VectorClass::operator*=(NN::type scalar){
    for (unsigned i = 0; i < numel; ++i)
        components[i] *= scalar;
    return *this;
}


VectorClass& VectorClass::operator/=(NN::type scalar){
    for (unsigned i = 0; i < numel; ++i)
        components[i] /= scalar;
    return *this;
}


void VectorClass::print(bool as_row) const{
    // print as ROW vector
    if ( as_row ) {
        for (unsigned i = 0; i < numel; ++i)
            std::cout << std::right << std::setw(NN::SPACING) << components[i];
        std::cout << std::endl;
    }
    // print as COLUMN vector
    else {
        for (unsigned i = 0; i < numel; ++i)
            std::cout << std::right << std::setw(NN::SPACING) << components[i] << "\n";
        std::cout << std::endl;
    }
}



void VectorClass::print(std::ofstream& out_stream) const{
    out_stream << numel << std::endl;
    for (unsigned i = 0; i < numel; ++i)
        out_stream << components[i] << ',';
    out_stream << '\n' << std::endl;
}


unsigned VectorClass::length(void) const{
    return numel;
}

void VectorClass::reset(void) const{
    for (unsigned i = 0; i < numel; ++i)
        components[i] = 0;
}
