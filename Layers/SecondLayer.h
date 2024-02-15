//
// Created by Pietro Manzoni on 17/02/22.
//

#ifndef RNN_P__LIBRARY_SECONDLAYER_H
#define RNN_P__LIBRARY_SECONDLAYER_H


#include "FirstLayer.h"

class SecondLayer : public FirstLayer {

public:

    /// Constructor of the object, which is a FirstLayer without autoregressive component
    /**
     * @param INPUT_SZ_:            number of pure \a exogenous inputs
     * @param OUTPUT_SZ_:           number of neurons of the layer
     * @param random_seed:          random seed
     */
    SecondLayer(unsigned INPUT_SZ_, unsigned OUTPUT_SZ_, unsigned random_seed) :
                FirstLayer(INPUT_SZ_, OUTPUT_SZ_, {}, random_seed) {}

};


#endif //RNN_P__LIBRARY_SECONDLAYER_H
