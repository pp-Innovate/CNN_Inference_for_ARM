// =====================================================================================
// 
//       Filename:  Vector.h
// 
//    Description:  Vector (1-D tensor) derives from the TensorBase base class
// 
//        Version:  1.0
//        Created:  11/06/17 15:31:57
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Peng Peng (pp), 1631552pp@tongji.edu.cn
//        Company:  Tongji University, Shanghai, China
// 
// =====================================================================================

#ifndef __CI4A_VECTOR_H
#define __CI4A_VECTOR_H

#include <cassert>

#include "tensor/TensorBase.h"

// =====================================================================================
//        Class:  Vector
//  Description:  1-D tensor derives from the TensorBase base class
// =====================================================================================
template <typename Type>
class Vector : public TensorBase<Type>
{

private:

protected:

public:

    // ===  FUNCTION  ======================================================================
    //         Name:  Vector
    //  Description:  Constructor
    // =====================================================================================
    Vector(uint16_t length, int8_t fractional_length) : TensorBase<Type>(length, 1, 1, 1, fractional_length)
    {

    }

    // ===  FUNCTION  ======================================================================
    //         Name:  allocate
    //  Description:  Dynamic memory allocation.
    //                Implementation of TensorBase's pure virtual method.
    // =====================================================================================
    virtual void allocate(void)
    {
        TensorBase<Type>::_data = new Type[TensorBase<Type>::_total_size];

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

        TensorBase<Type>::_is_allocated = false;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  dimension
    //  Description:  Return the size of specified dimension.
    //                The mandatory value of dim is 0.
    //                Implementation of TensorBase's pure virtual method.
    // =====================================================================================
    virtual uint16_t dimension(uint8_t dim) const
    {
        assert(dim < 1);

        return TensorBase<Type>::_dimension[dim];
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  set_dimension
    //  Description:  Set the size of specified dimension and update the total size.
    //                The mandatory value of dim is 0.
    //                Implementation of TensorBase's pure virtual method.
    // =====================================================================================
    virtual void set_dimension(uint8_t dim, uint16_t size)
    {
        assert(dim < 1);

        TensorBase<Type>::_dimension[dim] = size;
        TensorBase<Type>::update_total_size();
    }

}; // -----  end of class Vector  -----

#endif /* __CI4A_VECTOR_H */
