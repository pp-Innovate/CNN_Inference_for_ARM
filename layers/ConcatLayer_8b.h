// =====================================================================================
// 
//       Filename:  ConcatLayer_8b.h
// 
//    Description:  8-bit concat layer derives from the LayerBase base class
// 
//        Version:  1.0
//        Created:  11/09/17 11:03:51
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Peng Peng (pp), 1631552pp@tongji.edu.cn
//        Company:  Tongji University, Shanghai, China
// 
// =====================================================================================

#ifndef __CI4A_CONCATLAYER_8B_H
#define __CI4A_CONCATLAYER_8B_H

#include <vector>
#include <cassert>
#include <cstring>

#include "layers/LayerBase.h"
#include "tensor/Tensor3D.h"
#include "tensor/TensorMath.h"

// =====================================================================================
//        Class:  ConcatLayer_8b
//  Description:  8-bit concat layer derives from the LayerBase base class
// =====================================================================================
class ConcatLayer_8b : public LayerBase<int8_t, int8_t>
{

private:

    int8_t _fl_out;
    uint16_t _width, _height;
    uint16_t _channel_out;
    uint8_t _num_input;

protected:

public:

    // ===  FUNCTION  ======================================================================
    //         Name:  ConcatLayer_8b
    //  Description:  Constructor
    // =====================================================================================
    ConcatLayer_8b(int8_t fl_out) : LayerBase<int8_t, int8_t>()
    {
        _fl_out = fl_out;
        _width = 0;
        _height = 0;
        _channel_out = 0;
        _num_input = 0;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  initialize
    //  Description:  Implementation of the LayerBase's initialize pure virtual method
    // =====================================================================================
    virtual void initialize(void)
    {
        _width = LayerBase<int8_t, int8_t>::_input_tensor[0]->dimension(0);
        _height = LayerBase<int8_t, int8_t>::_input_tensor[0]->dimension(1);
        _num_input = LayerBase<int8_t, int8_t>::_input_tensor.size();

        for(uint8_t i = 1; i < _num_input; i++)
        {
            assert((LayerBase<int8_t, int8_t>::_input_tensor[i]->dimension(0)) == _width);
            assert((LayerBase<int8_t, int8_t>::_input_tensor[i]->dimension(1)) == _height);
        }

        for(uint8_t i = 0; i < _num_input; i++)
            _channel_out += LayerBase<int8_t, int8_t>::_input_tensor[i]->dimension(2);

        LayerBase<int8_t, int8_t>::_output_tensor = new Tensor3D<int8_t>(_width, _height, _channel_out, _fl_out);
        LayerBase<int8_t, int8_t>::_output_tensor->allocate();
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  forward
    //  Description:  Implementation of the forward propagation 
    // =====================================================================================
    virtual void forward(void)
    {
        int8_t *output_data = LayerBase<int8_t, int8_t>::_output_tensor->data();

        for(uint8_t i = 0; i < _num_input; i++)
        {
            if(LayerBase<int8_t, int8_t>::_input_tensor[i]->fractional_length() == _fl_out)
                memcpy(
                        output_data, 
                        LayerBase<int8_t, int8_t>::_input_tensor[i]->data(), 
                        LayerBase<int8_t, int8_t>::_input_tensor[i]->total_size() * sizeof(int8_t)
                );
            else
                TensorMath::shift_and_copy_8b(
                        LayerBase<int8_t, int8_t>::_input_tensor[i], 
                        output_data, 
                        _fl_out - LayerBase<int8_t, int8_t>::_input_tensor[i]->fractional_length()
                );

            output_data += LayerBase<int8_t, int8_t>::_input_tensor[i]->total_size();
        }
    }

}; // -----  end of class ConcatLayer_8b  -----

#endif /* __CI4A_CONCATLAYER_8B_H */
