// =====================================================================================
// 
//       Filename:  SoftmaxLayer_f.h
// 
//    Description:  Softmax layer derives from the LayerBase base class
// 
//        Version:  1.0
//        Created:  11/11/17 21:29:21
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Peng Peng (pp), 1631552pp@tongji.edu.cn
//        Company:  Tongji University, Shanghai, China
// 
// =====================================================================================

#ifndef __CI4A_SOFTMAXLAYER_F_H
#define __CI4A_SOFTMAXLAYER_F_H

#include <cmath>
#include <cassert>

#include "layers/LayerBase.h"
#include "tensor/Vector.h"

// =====================================================================================
//        Class:  SoftmaxLayer_f
//  Description:  Softmax layer derives from the LayerBase base layer
// =====================================================================================
class SoftmaxLayer_f : public LayerBase<float, float>
{

private:

    uint16_t _length;

    Vector<float> *_exp_tensor;

protected:

public:

    // ===  FUNCTION  ======================================================================
    //         Name:  SoftmaxLayer_f
    //  Description:  Constructor
    // =====================================================================================
    SoftmaxLayer_f(void) : LayerBase<float, float>()
    {
        _length = 0;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  initialize
    //  Description:  Implementation of the LayerBase's initialize pure virtual method
    // =====================================================================================
    virtual void initialize(void)
    {
        assert((LayerBase<float, float>::_input_tensor.size()) == 1);

        _length = LayerBase<float, float>::_input_tensor[0]->dimension(0);

        LayerBase<float, float>::_output_tensor = new Vector<float>(_length, 0);
        LayerBase<float, float>::_output_tensor->allocate();

        _exp_tensor = new Vector<float>(_length, 0);
        _exp_tensor->allocate();
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  forward
    //  Description:  Implementation of the forward propagation 
    // =====================================================================================
    virtual void forward(void)
    {
        float *__restrict__ input_data = LayerBase<float, float>::_input_tensor[0]->data();
        float *__restrict__ exp_data = _exp_tensor->data();
        float *__restrict__ output_data = LayerBase<float, float>::_output_tensor->data();

        float exp_sum = 0;

        for(uint16_t i = 0; i < _length; i++)
        {
            float x = input_data[i];

            uint8_t j = 1;
            float exp_x = 1;
            float increment = 1;

            while(abs(increment) > 1e-3)
            {
                increment *= x / j;
                exp_x += increment;
                j++;
            }

            exp_sum += exp_x;
            exp_data[i] = exp_x;
        }

        for(uint16_t i = 0; i < _length; i++)
            output_data[i] = exp_data[i] / exp_sum;
    }

}; // -----  end of class SoftmaxLayer_f  -----

#endif /* __CI4A_SOFTMAXLAYER_F_H */
