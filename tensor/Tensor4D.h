// =====================================================================================
// 
//       Filename:  Tensor4D.h
// 
//    Description:  4-D tensor derives from the TensorBase base class
// 
//        Version:  1.0
//        Created:  11/06/17 22:35:25
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Peng Peng (pp), 1631552pp@tongji.edu.cn
//        Company:  Tongji University, Shanghai, China
// 
// =====================================================================================

#ifndef __CI4A_TENSOR4D_H
#define __CI4A_TENSOR4D_H

#include <cassert>

#include "tensor/TensorBase.h"

// =====================================================================================
//        Class:  Tensor4D
//  Description:  4-D tensor derives from the TensorBase base class
// =====================================================================================
template <typename Type>
class Tensor4D : public TensorBase<Type>
{

private:

    Type **_address_table;

protected:

public:

    // ===  FUNCTION  ======================================================================
    //         Name:  Tensor4D
    //  Description:  Constructor
    // =====================================================================================
    Tensor4D(uint16_t width, uint16_t height, uint16_t channel, uint16_t batch, int8_t fractional_length) 
        : TensorBase<Type>(width, height, channel, batch, fractional_length)
    {
        _address_table = NULL;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  allocate
    //  Description:  Dynamic memory allocation.
    //                Implementation of TensorBase's pure virtual method.
    // =====================================================================================
    virtual void allocate(void)
    {
        TensorBase<Type>::_data = new Type[TensorBase<Type>::_total_size];

        _address_table = new Type *[TensorBase<Type>::_dimension[3]];

        size_t stride = TensorBase<Type>::_dimension[0] * TensorBase<Type>::_dimension[1] * TensorBase<Type>::_dimension[2];

        for(uint16_t i = 0; i < TensorBase<Type>::_dimension[3]; i++)
            _address_table[i] = TensorBase<Type>::_data + i * stride;

        TensorBase<Type>::_is_allocated = true;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  release
    //  Description:  Release the allocated memory.
    //                Implementation of TensorBase's pure virtual method.
    // =====================================================================================
    virtual void release(void)
    {
        delete[] TensorBase<Type>::_data;
        TensorBase<Type>::_data = NULL;

        delete[] _address_table;
        _address_table = NULL;

        TensorBase<Type>::_is_allocated = false;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  dimension
    //  Description:  Return the size of specified dimension.
    //                The value of dim should less than 4.
    //                Implementation of TensorBase's pure virtual method.
    // =====================================================================================
    virtual uint16_t dimension(uint8_t dim) const
    {
        assert(dim < 4);

        return TensorBase<Type>::_dimension[dim];
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  set_dimension
    //  Description:  Set the size of specified dimension and update the total size.
    //                The value of dim should less than 4.
    //                Implementation of TensorBase's pure virtual method.
    // =====================================================================================
    virtual void set_dimension(uint8_t dim, uint16_t size)
    {
        assert(dim < 4);

        TensorBase<Type>::_dimension[dim] = size;
        TensorBase<Type>::update_total_size();
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  ptr_batch
    //  Description:  Return a pointer to the start adderss of specified batch.
    //                The value of batch_index should in the range of [0, batch-1].
    // =====================================================================================
    Type *ptr_batch(uint16_t batch_index)
    {
        assert(batch_index < TensorBase<Type>::_dimension[3]);

        return _address_table[batch_index];
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  ~Tensor4D
    //  Description:  Destructor
    // =====================================================================================
    ~Tensor4D(void)
    {
        if(TensorBase<Type>::_is_allocated)
            release();
    }

}; // -----  end of class Tensor4D  -----

#endif /* __CI4A_TENSOR4D_H */
