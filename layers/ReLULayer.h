#ifndef __CI4A_RELULAYER_H
#define __CI4A_RELULAYER_H

#include <iostream>
#include <unistd.h>
#include <sys/time.h>

#include <arm_neon.h>

#include "tensor/Tensor.h"

using namespace std;

class ReLULayer
{

private:

    Tensor *_input_tensor;
    Tensor *_output_tensor;

public:

    /* Constructor */
    ReLULayer(Tensor *input_tensor)
    {
        /* Member variables initializaiton */
        _input_tensor = input_tensor;

        /* Create output tensor object */
        _output_tensor = new Tensor(
                _input_tensor->dimension(0), 
                _input_tensor->dimension(1), 
                _input_tensor->dimension(2), 
                _input_tensor->dimension(3), 
                _input_tensor->fractional_length()
            );
        _output_tensor->allocate();
    }

    /* Implementation of forward propagation */
    void forward(void)
    {
        int8_t *input_data = _input_tensor->data();
        int8_t *output_data = _output_tensor->data();

        size_t length_div_16 = _input_tensor->total_size() / 16;
        uint8_t length_mod_16 = _input_tensor->total_size() % 16;

        int8x16_t vec_in, vec_out;
        int8x16_t vec_zero = vdupq_n_s8(0);

        for(size_t i = 0; i < length_div_16; i++)
        {
            vec_in = vld1q_s8(input_data);
            vec_out = vmaxq_s8(vec_in, vec_zero);
            vst1q_s8(output_data, vec_out);

            input_data += 16;
            output_data += 16;
        }

        if(length_mod_16)
        {
            int8_t temp1[16] = {0};
            int8_t temp2[16] = {0};

            memcpy((void *)temp1, (void *)input_data, length_mod_16);

            vec_in = vld1q_s8(temp1);
            vec_out = vmaxq_s8(vec_in, vec_zero);
            vst1q_s8(temp2, vec_out);

            memcpy((void *)output_data, (void *)temp2, length_mod_16);
        }
    }

    /* Return a pointer to the output tensor */
    Tensor *ptr_output_tensor(void)
    {
        return _output_tensor;
    }

};

#endif /* __CI4A_RELULAYER_H */
