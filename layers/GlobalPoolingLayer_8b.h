// =====================================================================================
// 
//       Filename:  GlobalPoolingLayer_8b.h
// 
//    Description:  Global pooling layer derivers from the LayerBase base class
// 
//        Version:  1.0
//        Created:  11/11/17 16:36:44
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Peng Peng (pp), 1631552pp@tongji.edu.cn
//        Company:  Tongji University, Shanghai, China
// 
// =====================================================================================

#ifndef __CI4A_GLOBALPOOLINGLAYER_8B_H
#define __CI4A_GLOBALPOOLINGLAYER_8B_H

#include <cassert>

#include "layers/LayerBase.h"
#include "tensor/Tensor3D.h"
#include "tensor/Vector.h"

// =====================================================================================
//        Class:  GlobalPoolingLayer_8b
//  Description:  Global pooling layer derives from the LayerBase base class
// =====================================================================================
class GlobalPoolingLayer_8b : public LayerBase<int8_t, float>
{

private:

    uint32_t _featuremap_size;
    uint16_t _featuremap_num;

protected:

public:

    // ===  FUNCTION  ======================================================================
    //         Name:  GlobalPoolingLayer_8b
    //  Description:  Constructor
    // =====================================================================================
    GlobalPoolingLayer_8b(void) : LayerBase<int8_t, float>()
    {
        _featuremap_num = 0;
        _featuremap_size = 0;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  initialize
    //  Description:  Implementation of the LayerBase's initialize pure virtual method
    // =====================================================================================
    virtual void initialize(void)
    {
        assert((LayerBase<int8_t, float>::_input_tensor.size()) == 1);

        _featuremap_num = LayerBase<int8_t, float>::_input_tensor[0]->dimension(2);
        _featuremap_size = LayerBase<int8_t, float>::_input_tensor[0]->dimension(0) * LayerBase<int8_t, float>::_input_tensor[0]->dimension(1);

        LayerBase<int8_t, float>::_output_tensor = new Vector<float>(_featuremap_num, 0);
        LayerBase<int8_t, float>::_output_tensor->allocate();
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  forward
    //  Description:  Implementation of the forward propagation 
    // =====================================================================================
    virtual void forward(void)
    {
        int8_t *__restrict__ input_data = LayerBase<int8_t, float>::_input_tensor[0]->data();
        float *__restrict__ output_data = LayerBase<int8_t, float>::_output_tensor->data();

        for(uint16_t i = 0; i < _featuremap_num; i++)
        {
            int32_t sum = 0;

            for(uint32_t j = 0; j < _featuremap_size; j++)
                sum += *(input_data++);

            *(output_data++) = (float)sum / _featuremap_size;
        }
    }

}; // -----  end of class GlobalPoolingLayer_8b  -----

#endif /* __CI4A_GLOBALPOOLINGLAYER_8B_H */
