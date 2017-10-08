#ifndef __CI4A_GLOBALPOOLINGLAYER_H
#define __CI4A_GLOBALPOOLINGLAYER_H

#include <iostream>
#include <cassert>
#include <unistd.h>
#include <sys/time.h>

#include <arm_neon.h>

#include "tensor/Tensor.h"

using namespace std;

class GlobalPoolingLayer
{

private:

    uint32_t _featuremap_size;
    uint16_t _featuremap_num;

    Tensor *_input_tensor;
    Tensor *_output_tensor;

public:

    /* Constructor */
    GlobalPoolingLayer(Tensor *input_tensor)
    {
        /* Member variables initializaiton */
        _input_tensor = input_tensor;
        _featuremap_size = _input_tensor->dimension(0) * _input_tensor->dimension(1);
        _featuremap_num = _input_tensor->dimension(2);

        /* Create output tensor object */
        _output_tensor = new Tensor(_featuremap_num, _input_tensor->fractional_length());
        _output_tensor->allocate();
    }

    /* Implementation of forward propagation */
    void forward(void)
    {
        int8_t *input_data = _input_tensor->data();
        int8_t *output_data = _output_tensor->data();

        for(uint16_t i = 0; i < _featuremap_num; i++)
        {
            int32_t sum = 0;

            for(uint32_t j = 0; j < _featuremap_size; j++)
                sum += *(input_data++);

            *(output_data++) = (int8_t)((float)sum / _featuremap_size);
        }
    }

    /* Return a pointer to the output tensor */
    Tensor *ptr_output_tensor(void)
    {
        return _output_tensor;
    }
};

#endif /* __CI4A_GLOBALPOOLINGLAYER_H */
