// =====================================================================================
// 
//       Filename:  ReLULayer_8b.h
// 
//    Description:  8-bit ReLU layer derives from the LayerBase base class
// 
//        Version:  1.0
//        Created:  11/07/17 15:28:52
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Peng Peng (pp), 1631552pp@tongji.edu.cn
//        Company:  Tongji University, Shanghai, China
// 
// =====================================================================================

#ifndef __CI4A_RELULAYER_8B_H
#define __CI4A_RELULAYER_8B_H

#include <vector>
#include <cassert>

#include <arm_neon.h>

#include "layers/LayerBase.h"
#include "tensor/Tensor3D.h"

// =====================================================================================
//        Class:  ReLULayer_8b
//  Description:  8-bit ReLU layer derives from the LayerBase base class
// =====================================================================================
class ReLULayer_8b : public LayerBase<int8_t, int8_t>
{

private:

protected:

public:

    // ===  FUNCTION  ======================================================================
    //         Name:  initialize
    //  Description:  Implementation of the LayerBase's initialize pure virtual method
    // =====================================================================================
    virtual void initialize(void)
    {
        assert((LayerBase<int8_t, int8_t>::_input_tensor.size()) == 1);

        TensorBase<int8_t> *input_tensor = LayerBase<int8_t, int8_t>::_input_tensor[0];

        LayerBase<int8_t, int8_t>::_output_tensor = new Tensor3D<int8_t>(
                input_tensor->dimension(0), 
                input_tensor->dimension(1), 
                input_tensor->dimension(2), 
                input_tensor->fractional_length()
        );
        LayerBase<int8_t, int8_t>::_output_tensor->allocate();
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  forward
    //  Description:  Implementation of the forward propagation 
    // =====================================================================================
    virtual void forward(void)
    {
        int8_t *input_data = LayerBase<int8_t, int8_t>::_input_tensor[0]->data();
        int8_t *output_data = LayerBase<int8_t, int8_t>::_output_tensor->data();

        size_t length_div_16 = LayerBase<int8_t, int8_t>::_input_tensor[0]->total_size() / 16;
        uint8_t length_mod_16 = LayerBase<int8_t, int8_t>::_input_tensor[0]->total_size() % 16;

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

        for(uint8_t i = 0; i < length_mod_16; i++)
            vec_in = vsetq_lane_s8(*input_data++, vec_in, i);

        vec_out = vmaxq_s8(vec_in, vec_zero);

        for(uint8_t i = 0; i < length_mod_16; i++)
            *output_data++ = vgetq_lane_s8(vec_out, i);
    }

}; // -----  end of class ReLULayer_8b  -----

#endif /* __CI4A_RELULAYER_8B_H */
