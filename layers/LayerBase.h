// =====================================================================================
// 
//       Filename:  LayerBase.h
// 
//    Description:  The abstract base class for any deep neural network layers
// 
//        Version:  1.0
//        Created:  11/07/17 14:12:29
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Peng Peng (pp), 1631552pp@tongji.edu.cn
//        Company:  Tongji University, Shanghai, China
// 
// =====================================================================================

#ifndef __CI4A_LAYERBASE_H
#define __CI4A_LAYERBASE_H

#include <vector>

#include "tensor/TensorBase.h"

// =====================================================================================
//        Class:  LayerBase
//  Description:  The abstract base class for any deep neural network layers
// =====================================================================================
template <typename InType, typename OutType>
class LayerBase
{

private:

protected:

    std::vector< TensorBase<InType> *> _input_tensor;
    TensorBase<OutType> *_output_tensor;

public:

    // ===  FUNCTION  ======================================================================
    //         Name:  connect
    //  Description:  Connect one of the input tensor(s) to this layer
    // =====================================================================================
    void connect(TensorBase<InType> *input_tensor)
    {
        _input_tensor.push_back(input_tensor);
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  initialize
    //  Description:  Initialize this layer after connect
    // =====================================================================================
    virtual void initialize(void) = 0;

    // ===  FUNCTION  ======================================================================
    //         Name:  forward
    //  Description:  Forward propagation
    // =====================================================================================
    virtual void forward(void) = 0;

    // ===  FUNCTION  ======================================================================
    //         Name:  ptr_output_tensor
    //  Description:  Return a pointer to the output tensor
    // =====================================================================================
    TensorBase<OutType> *ptr_output_tensor(void) const
    {
        return _output_tensor;
    }

}; // -----  end of class LayerBase  -----

#endif /* __CI4A_LAYERBASE_H */
