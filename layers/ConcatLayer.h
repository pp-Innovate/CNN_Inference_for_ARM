#ifndef __CI4A_CONCATLAYER_H
#define __CI4A_CONCATLAYER_H

#include <iostream>
#include <cassert>
#include <unistd.h>
#include <sys/time.h>

#include <arm_neon.h>

#include "tensor/Tensor.h"
#include "tensor/TensorMath.h"

using namespace std;

class ConcatLayer
{

private:

    uint16_t _width, _height, _channel_out;
    int8_t _fl_in_1, _fl_in_2, _fl_out;

    Tensor *_input_tensor_1, *_input_tensor_2;
    Tensor *_output_tensor;

public:

    /* Constructor */
    ConcatLayer(Tensor *input_tensor_1, Tensor *input_tensor_2, int8_t fl_out)
    {
        /* Member variables initializaiton */
        _input_tensor_1 = input_tensor_1;
        _input_tensor_2 = input_tensor_2;
        _fl_in_1 = _input_tensor_1->fractional_length();
        _fl_in_2 = _input_tensor_2->fractional_length();
        _fl_out = fl_out;

        assert(_input_tensor_1->dimension(0) == _input_tensor_2->dimension(0));
        assert(_input_tensor_1->dimension(1) == _input_tensor_2->dimension(1));

        _width = _input_tensor_1->dimension(0);
        _height = _input_tensor_1->dimension(1);
        _channel_out = _input_tensor_1->dimension(2) + _input_tensor_2->dimension(2);

        /* Create the output tensor object */
        _output_tensor = new Tensor(_width, _height, _channel_out, _fl_out);
        _output_tensor->allocate();
    }
    
    /* Implementation of forward propagation */
    void forward(void)
    {
        if(_fl_in_1 == _fl_out)
            memcpy(_output_tensor->data(), _input_tensor_1->data(), _input_tensor_1->total_size());
        else
            TensorMath::shift_and_copy(_input_tensor_1, _output_tensor->data(), _fl_out - _fl_in_1);

        if(_fl_in_2 == _fl_out)
            memcpy(_output_tensor->data() + _input_tensor_1->total_size(), _input_tensor_2->data(), _input_tensor_2->total_size());
        else
            TensorMath::shift_and_copy(_input_tensor_2, _output_tensor->data() + _input_tensor_1->total_size(), _fl_out - _fl_in_2);
    }

    /* Return a pointer to the output tensor */
    Tensor *ptr_output_tensor(void)
    {
        return _output_tensor;
    }
};

#endif /* __CI4A_CONCATLAYER_ */
