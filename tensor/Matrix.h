// =====================================================================================
// 
//       Filename:  Matrix.h
// 
//    Description:  Matrix (2-D tensor) derives from the TensorBase base class
// 
//        Version:  1.0
//        Created:  11/06/17 20:45:24
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Peng Peng (pp), 1631552pp@tongji.edu.cn
//        Company:  Tongji University, Shanghai, China
// 
// =====================================================================================

#ifndef __CI4A_MATRIX_H
#define __CI4A_MATRIX_H

#include <cassert>

#include "tensor/TensorBase.h"

// =====================================================================================
//        Class:  Matrix
//  Description:  2-D tensor derives from the TensorBase base class
// =====================================================================================
template <typename Type>
class Matrix : public TensorBase<Type>
{

private:

    Type **_address_table;

protected:

public:

    // ===  FUNCTION  ======================================================================
    //         Name:  Matrix
    //  Description:  Constructor
    // =====================================================================================
    Matrix(uint16_t width, uint16_t height, int8_t fractional_length) : TensorBase<Type>(width, height, 1, 1, fractional_length)
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

        _address_table = new Type *[TensorBase<Type>::_dimension[1]];

        for(uint16_t i = 0; i < TensorBase<Type>::_dimension[1]; i++)
            _address_table[i] = TensorBase<Type>::_data + i * TensorBase<Type>::_dimension[0];

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
    //                The value of dim should less than 2.
    //                Implementation of TensorBase's pure virtual method.
    // =====================================================================================
    virtual uint16_t dimension(uint8_t dim) const
    {
        assert(dim < 2);

        return TensorBase<Type>::_dimension[dim];
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  set_dimension
    //  Description:  Set the size of specified dimension and update the total size.
    //                The value of dim should less than 2.
    //                Implementation of TensorBase's pure virtual method.
    // =====================================================================================
    virtual void set_dimension(uint8_t dim, uint16_t size)
    {
        assert(dim < 2);

        TensorBase<Type>::_dimension[dim] = size;
        TensorBase<Type>::update_total_size();
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  ptr_row
    //  Description:  Return a pointer to the start adderss of specified row.
    //                The value of row_index should in the range of [0, height-1].
    // =====================================================================================
    Type *ptr_row(uint16_t row_index)
    {
        assert(row_index < TensorBase<Type>::_dimension[1]);

        return _address_table[row_index];
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  ~Matrix
    //  Description:  Destructor
    // =====================================================================================
    ~Matrix(void)
    {
        if(TensorBase<Type>::_is_allocated)
            release();
    }

}; // -----  end of class Matrix  -----

#endif /* __CI4A_MATRIX_H */
