#ifndef __CI4A_CONVOLUTIONALLAYER_H
#define __CI4A_CONVOLUTIONALLAYER_H

#include <iostream>
#include <cassert>
#include <unistd.h>
#include <sys/time.h>

#include <arm_neon.h>

#include "tensor/Tensor.h"
#include "tensor/TensorMath.h"

using namespace std;

class ConvolutionalLayer
{

private:

    uint16_t _width, _height, _channel_in, _channel_out;
    uint8_t _kernel_size, _stride, _pad;
    int8_t _fl_in, _fl_out, _fl_param;
    uint16_t _width_out, _height_out;

    uint32_t _mat_cols_old, _mat_cols;
    uint8_t _insert_zeros;
    uint16_t _mat_rows;

    Tensor *_input_tensor;
    Tensor *_im2col_tensor;
    Tensor *_weights, *_biases;
    Tensor *_output_tensor;
    Tensor *_weight_matrix;

    void biases_preprocess(void)
    {
        int8x16_t vec_shift_bits = vdupq_n_s8(_fl_in);
        int8x16_t vec_data;

        int8_t *data = _biases->data();

        for(uint16_t i = 0; i < _channel_out / 16; i++)
        {
            vec_data = vld1q_s8(data);
            vec_data = vrshlq_s8(vec_data, vec_shift_bits);
            vst1q_s8(data, vec_data);

            data += 16;
        }

        //for(uint8_t i = 0; i < _channel_out % 16; i++)
        //    vec_data = vsetq_lane_s8(data[i], vec_data, i);

        //vec_data = vrshlq_s8(vec_data, vec_shift_bits);

        //for(uint8_t i = 0; i < _channel_out % 16; i++)
        //    data[i] = vgetq_lane_s8(vec_data, i);
        
        int8_t temp[16];

        for(uint8_t i = 0; i < _channel_out % 16; i++)
            temp[i] = data[i];

        vec_data = vld1q_s8(temp);
        vec_data = vrshlq_s8(vec_data, vec_shift_bits);
        vst1q_s8(temp, vec_data);

        for(uint8_t i = 0; i < _channel_out % 16; i++)
            data[i] = temp[i];
    }

    void weights2matrix(void)
    {
        int8_t *tensor_data = _weights->data();
        int8_t *matrix_data = _weight_matrix->data();

        for(uint16_t i = 0; i < _channel_out; i++)
        {
            for(uint16_t j = 0; j < _mat_cols_old; j++)
                *(matrix_data++) = *(tensor_data++);
            for(uint8_t j = 0; j < _insert_zeros; j++)
                *(matrix_data++) = 0;
        }
    }

    void im2col_1x1(void)
    {
        int8_t *data_col = _im2col_tensor->data();

        if(!_input_tensor)
            for(uint16_t i = 0; i < _width * _height; i++)
                for(uint16_t j = 0; j < _channel_in; j++)
                    *(data_col++) = *(_input_tensor->ptr_outermost_dimension(j) + i);
        else
        {
            for(uint16_t i = 0; i < _width * _height; i++)
            {
                for(uint16_t j = 0; j < _channel_in; j++)
                    *(data_col++) = *(_input_tensor->ptr_outermost_dimension(j) + i);
                for(uint16_t j = 0; j < _insert_zeros; j++)
                    *(data_col++) = 0;
            }
        }

    }

    void im2col_3x3(void)
    {
        int32_t height_m_1 = _height - 1;
        int32_t width_m_1 = _width - 1;
        int32_t width_m_2 = _width - 2;

        int8_t *data_col = _im2col_tensor->data();

        int32_t input_row_start = -_pad;
        for(uint16_t output_row = 0; output_row < _height_out; output_row++)
        {
            int32_t input_col_start = -_pad;
            for(uint16_t output_col = 0; output_col < _width_out; output_col++)
            {
                for(uint16_t channel = 0; channel < _channel_in; channel++)
                {
                    int8_t *data_im = _input_tensor->ptr_outermost_dimension(channel);
                    for(uint8_t kernel_row = 0; kernel_row < 3; kernel_row++)       //This loop can be unrolled if using flag "-O3"
                    {
                        int32_t input_row = input_row_start + kernel_row;
                        if(static_cast<unsigned>(input_row) > static_cast<unsigned>(height_m_1))
                        {
                            *(data_col++) = 0;
                            *(data_col++) = 0;
                            *(data_col++) = 0;
                        }
                        else
                        {
                            int32_t input_col = input_row * _width + input_col_start;

                            if(input_col_start > -1 && input_col_start < width_m_2)
                            {
                                *(data_col++) = data_im[input_col];
                                *(data_col++) = data_im[input_col + 1];
                                *(data_col++) = data_im[input_col + 2];
                            }
                            else if(input_col_start == -2)
                            {
                                *(data_col++) = 0;
                                *(data_col++) = 0;
                                *(data_col++) = data_im[input_col + 2];
                            }
                            else if(input_col_start == -1)
                            {
                                *(data_col++) = 0;
                                *(data_col++) = data_im[input_col + 1];
                                *(data_col++) = data_im[input_col + 2];
                            }
                            else if(input_col_start == width_m_2)
                            {
                                *(data_col++) = data_im[input_col];
                                *(data_col++) = data_im[input_col + 1];
                                *(data_col++) = 0;
                            }
                            else if(input_col_start == width_m_1)
                            {
                                *(data_col++) = data_im[input_col];
                                *(data_col++) = 0;
                                *(data_col++) = 0;
                            }
                            else
                            {
                                *(data_col++) = 0;
                                *(data_col++) = 0;
                                *(data_col++) = 0;
                            }
                        }
                    }
                }
                input_col_start += _stride;

                for(uint8_t i = 0; i < _insert_zeros; i++)
                    *(data_col++) = 0;
            }
            input_row_start += _stride;
        }
    }

public:

    /* Constructor */
    ConvolutionalLayer(
            Tensor *input_tensor, 
            uint8_t kernel_size, uint16_t channel_out, 
            uint8_t stride, uint8_t pad, 
            int8_t fl_param, int8_t fl_out
        )
    {
        /* Member variables initializaiton */
        _input_tensor = input_tensor;
        _width = _input_tensor->dimension(0);
        _height = _input_tensor->dimension(1);
        _channel_in = _input_tensor->dimension(2);
        _channel_out = channel_out;
        _kernel_size = kernel_size;
        _stride = stride;
        _pad = pad;
        _fl_in = _input_tensor->fractional_length();
        _fl_out = fl_out;
        _fl_param = fl_param;
        _width_out = (_width + (_pad << 1) - _kernel_size) / _stride + 1;
        _height_out = (_height + (_pad << 1) - _kernel_size) / _stride + 1;
        _mat_cols_old = _kernel_size * _kernel_size * _channel_in;
        _mat_cols = (_mat_cols_old % 8) ? ((_mat_cols_old / 8 + 1) * 8) : _mat_cols_old;
        _insert_zeros = _mat_cols - _mat_cols_old;
        _mat_rows = (_channel_out % 4) ? (_channel_out / 4 + 1) : (_channel_out / 4);

        /* Create weights and biases tensor objects */
        _weights = new Tensor(_kernel_size, _kernel_size, _channel_in, _channel_out, _fl_param);
        _weights->allocate();
        _biases = new Tensor(_channel_out, _fl_param);
        _biases->allocate();

        /* Create weight matrix object but do not allocate memory */
        _weight_matrix = new Tensor(_mat_cols, _channel_out, _fl_param);

        /* Create output tensor object */
        _output_tensor = new Tensor(_width_out, _height_out, channel_out, _fl_out);
        _output_tensor->allocate();

        /* Create im2col tensor object */
        _im2col_tensor = new Tensor(_mat_cols, _width_out * _height_out, _fl_in);
        _im2col_tensor->allocate();
    }

    /* Implementation of forward propagation */
    void forward(void)
    {
        if(!_weight_matrix->is_allocated())
        {
            _weight_matrix->allocate();
            weights2matrix();
        }

        if(_kernel_size == 1)
        {
            im2col_1x1();
            TensorMath::A_mul_Btranspose(_weight_matrix, _im2col_tensor, _biases, _output_tensor);

        }
        else if(_kernel_size == 3)
        {
            im2col_3x3();
            TensorMath::A_mul_Btranspose(_weight_matrix, _im2col_tensor, _biases, _output_tensor);
        }
    }

    /* Return a pointer to the output tensor */
    Tensor *ptr_output_tensor(void)
    {
        return _output_tensor;
    }

    /* Return a pointer to the weight tensor */
    Tensor *ptr_weights_tensor(void)
    {
        return _weights;
    }

    /* Return a pointer to the bias tensor */
    Tensor *ptr_biases_tensor(void)
    {
        return _biases;
    }

};

#endif /* __CI4A_CONVOLUTIONALLAYER_H */
