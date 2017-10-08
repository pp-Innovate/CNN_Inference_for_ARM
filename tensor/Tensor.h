#ifndef __CI4A_TENSOR_H
#define __CI4A_TENSOR_H

#include <iostream>
#include <iomanip>

#include <arm_neon.h>

using namespace std;

class Tensor
{

private:

    int8_t *_data = NULL;

    uint16_t _dimension[4] = {0, 0, 0, 0};
    size_t _total_size = 0;

    int8_t _fractional_length = 0;

    bool _is_allocated = false;

    void calcTotalSize(void)
    {
        _total_size = _dimension[0] * _dimension[1] * _dimension[2] * _dimension[3];
    }

public:

    /* Default constructor */
    Tensor(void) {}

    /* Constructor for 1-D tensor (vector) */
    Tensor(uint16_t length, int8_t fractional_length)
    {
        _dimension[0] = length;
        _dimension[1] = 1;
        _dimension[2] = 1;
        _dimension[3] = 1;

        calcTotalSize();

        _fractional_length = fractional_length;
    }

    /* Constructor for 2-D tensor (matrix) */
    Tensor(uint16_t width, uint16_t height, int8_t fractional_length)
    {
        _dimension[0] = width;
        _dimension[1] = height;
        _dimension[2] = 1;
        _dimension[3] = 1;

        calcTotalSize();

        _fractional_length = fractional_length;
    }

    /* Constructor for 2-D tensor with pre-allocated memory */
    Tensor (uint16_t width, uint16_t height, int8_t fractional_length, int8_t *data)
    {
        _dimension[0] = width;
        _dimension[1] = height;
        _dimension[2] = 1;
        _dimension[3] = 1;

        calcTotalSize();

        _fractional_length = fractional_length;

        _data = data;
    }

    /* Constructor for 3-D tensor (i.e. input/outpu tensor of convolutional layers) */
    Tensor(uint16_t width, uint16_t height, uint16_t channel, int8_t fractional_length)
    {
        _dimension[0] = width;
        _dimension[1] = height;
        _dimension[2] = channel;
        _dimension[3] = 1;

        calcTotalSize();

        _fractional_length = fractional_length;
    }

    /* Constructor for 4-D tensor (i.e. weight tensor of convolutional layers */
    Tensor(uint16_t width, uint16_t height, uint16_t channel, uint16_t batch, int8_t fractional_length)
    {
        _dimension[0] = width;
        _dimension[1] = height;
        _dimension[2] = channel;
        _dimension[3] = batch;

        calcTotalSize();

        _fractional_length = fractional_length;
    }

    /* Allocate size specified by _total_size of CPU memory */
    void allocate(void)
    {
        _data = new int8_t[_total_size];

        _is_allocated = true;
    }

    /* Return the memory allocate status */
    bool is_allocated(void)
    {
        return _is_allocated;
    }

    /* Free the allocated CPU memory */
    void release(void)
    {
        delete[] _data;
        _data = NULL;

        _is_allocated = false;
    }

    /* Return a pointer to the allocated cpu data */
    int8_t *data(void)
    {
        return _data;
    }

    /* Return the number of elements held by this tensor */
    size_t total_size(void)
    {
        return _total_size;
    }

    /* Return the size of specified dimension */
    uint16_t dimension(uint8_t dim)
    {
        return _dimension[dim];
    }

    /* Set the size of specified dimension */
    void set_dimension(uint8_t dim, uint16_t size)
    {
        _dimension[dim] = size;

        calcTotalSize();
    }

    /* Return the fractional length */
    int8_t fractional_length(void)
    {
        return _fractional_length;
    }

    /* Set the fractional length */
    void set_fractional_length(int8_t fractional_length)
    {
        _fractional_length = fractional_length;
    }

    /* Print the tensor to stdout */
    void print(void)
    {
        int8_t *data = _data;

        for(uint16_t i = 0; i < _dimension[3]; i++)
        {
            for(uint16_t j = 0; j < _dimension[2]; j++)
            {
                for(uint16_t k = 0; k < _dimension[1]; k++)
                {
                    for(uint16_t l = 0; l < _dimension[0]; l++)
                    {
                        cout << setw(5) << (int)*(data++);
                    }
                    cout << endl;
                }
                cout << endl;
            }
            cout << endl << endl;
        }
    }

    /* Destructor */
    ~Tensor(void)
    {
        if(_is_allocated)
            release();
    }
};

#endif /* __CI4A_TENSOR_H */
