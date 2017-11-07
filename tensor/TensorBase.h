// =====================================================================================
// 
//       Filename:  TensorBase.h
// 
//    Description:  The abstract base class for array-like data structures
// 
//        Version:  1.0
//        Created:  11/05/17 13:10:25
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Peng Peng (pp), 1631552pp@tongji.edu.cn
//        Company:  Tongji University, Shanghai, China
// 
// =====================================================================================
 
#ifndef __CI4A_TENSORBASE_H
#define __CI4A_TENSORBASE_H

#include <iostream>
#include <iomanip>
#include <typeinfo>
#include <cassert>

using namespace std;

// =====================================================================================
//        Class:  TensorBase
//  Description:  Abstract base class for vector, matrix, 3-D tensor and 4-D tensor
// =====================================================================================
template <typename Type>
class TensorBase
{

private:

    int8_t _fractional_length;

protected:

    Type *_data;

    uint16_t _dimension[4];
    size_t _total_size;

    bool _is_allocated;

    // ===  FUNCTION  ======================================================================
    //         Name:  update_total_size
    //  Description:  Calculate the number of elements according to dimensional information
    // =====================================================================================
    void update_total_size(void)
    {
        _total_size = _dimension[0] * _dimension[1] * _dimension[2] * _dimension[3];
    }

public:

    // ===  FUNCTION  ======================================================================
    //         Name:  TensorBase
    //  Description:  Constructor
    // =====================================================================================
    TensorBase(uint16_t dim0, uint16_t dim1 = 1, uint16_t dim2 = 1, uint16_t dim3 = 1, int8_t fractional_length = 0)
    {
        _data = NULL;

        _dimension[0] = dim0;
        _dimension[1] = dim1;
        _dimension[2] = dim2;
        _dimension[3] = dim3;

        update_total_size();

        _fractional_length = fractional_length;

        _is_allocated = false;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  allocate
    //  Description:  Dynamic memory allocation
    // =====================================================================================
    virtual void allocate(void) = 0;

    // ===  FUNCTION  ======================================================================
    //         Name:  release
    //  Description:  Release the allocated memory
    // =====================================================================================
    virtual void release(void) = 0;

    // ===  FUNCTION  ======================================================================
    //         Name:  is_allocated
    //  Description:  Return the memory allocation status
    // =====================================================================================
    bool is_allocated(void) const
    {
        return _is_allocated;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  data
    //  Description:  Return a pointer to start address of allocated memory
    // =====================================================================================
    Type *data(void) const
    {
        return _data;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  total_size
    //  Description:  Return the number of elements held by this tensor
    // =====================================================================================
    size_t total_size(void) const
    {
        return _total_size;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  dimension
    //  Description:  Return the size of a specified dimension
    // =====================================================================================
    virtual uint16_t dimension(uint8_t dim) const = 0;

    // ===  FUNCTION  ======================================================================
    //         Name:  set_dimension
    //  Description:  Set the size of a specified dimension and update the total size
    // =====================================================================================
    virtual void set_dimension(uint8_t dim, uint16_t size) = 0;

    // ===  FUNCTION  ======================================================================
    //         Name:  fractional_length
    //  Description:  Return the fractional length.
    //                Notice that this member is effective only for fixed-point types.
    // =====================================================================================
    int8_t fractional_length(void) const
    {
        return _fractional_length;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  set_fractional_length
    //  Description:  Set the fractional length.
    //                Notice that this member is effective only for fixed-point types.
    // =====================================================================================
    void set_fractional_length(int8_t fractional_length)
    {
        _fractional_length = fractional_length;
    }

    // ===  FUNCTION  ======================================================================
    //         Name:  print
    //  Description:  Print the tensor to stdout
    // =====================================================================================
    void print(void) const
    {
        Type *data = _data;

        if(typeid(Type) == typeid(float) || typeid(Type) == typeid(double))
        {
            for(uint16_t i = 0; i < _dimension[3]; i++)
            {
                for(uint16_t j = 0; j < _dimension[2]; j++)
                {
                    for(uint16_t k = 0; k < _dimension[1]; k++)
                    {
                        for(uint16_t l = 0; l < _dimension[0]; l++)
                        {
                            cout << setprecision(2) << *data++ << "  ";
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl << endl;
            }
        
        }

        else if(typeid(Type) == typeid(int8_t) || typeid(Type) == typeid(uint8_t))
        {
            for(uint16_t i = 0; i < _dimension[3]; i++)
            {
                for(uint16_t j = 0; j < _dimension[2]; j++)
                {
                    for(uint16_t k = 0; k < _dimension[1]; k++)
                    {
                        for(uint16_t l = 0; l < _dimension[0]; l++)
                        {
                            cout << setw(5) << (int)(*data++);
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl << endl;
            }
        }

        else if(typeid(Type) == typeid(int16_t) || typeid(Type) == typeid(uint16_t))
        {
            for(uint16_t i = 0; i < _dimension[3]; i++)
            {
                for(uint16_t j = 0; j < _dimension[2]; j++)
                {
                    for(uint16_t k = 0; k < _dimension[1]; k++)
                    {
                        for(uint16_t l = 0; l < _dimension[0]; l++)
                        {
                            cout << setw(7) << *data++;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl << endl;
            }
        }

        else
        {
            for(uint16_t i = 0; i < _dimension[3]; i++)
            {
                for(uint16_t j = 0; j < _dimension[2]; j++)
                {
                    for(uint16_t k = 0; k < _dimension[1]; k++)
                    {
                        for(uint16_t l = 0; l < _dimension[0]; l++)
                        {
                            cout << *data++ << "  ";
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl << endl;
            }
        }
    }

}; // -----  end of class TensorBase  -----

#endif /* __CI4A_TENSORBASE_H */
