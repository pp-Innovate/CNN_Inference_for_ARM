#ifndef __CI4A_TENSORMATH_H
#define __CI4A_TENSORMATH_H

#include <iostream>
#include <cassert>

#include <arm_neon.h>

#include "tensor/Tensor.h"

using namespace std;

class TensorMath
{

public:

    static void A_mul_Btranspose_pack4(Tensor *A, Tensor *B, Tensor *biases, Tensor *C)
    {
        uint16_t A_rows = A->dimension(1), A_cols = A->dimension(0);
        uint16_t B_rows = B->dimension(1), B_cols = B->dimension(0);
        uint32_t B_cols_mul_2 = B_cols * 2;
        uint32_t B_cols_mul_3 = B_cols * 3;

        assert(A_cols == B_cols);
        assert(biases->dimension(0) == A_rows);

        uint16_t B_cols_div_8 = B_cols / 8;
        int64x2_t vec_shift_bits_1 = vdupq_n_s64(-B->fractional_length());
        int64x2_t vec_shift_bits_2 = vdupq_n_s64(C->fractional_length() - A->fractional_length());

        int8_t *C_data = C->data();

        for(uint16_t i = 0; i < A_rows; i++)
        {
            int64x2_t vec_bias = vdupq_n_s64(*(biases->data() + i));
            int8_t *A_start = A->data() + i * A_cols;

            for(uint16_t j = 0; j < B_rows; j += 4)
            {
                int8_t *A_data = A_start;
                int8_t *B_data = B->data() + j * B_cols;

                int32x4x4_t vecs_acc1;
                vecs_acc1.val[0] = vdupq_n_s32(0);
                vecs_acc1.val[1] = vdupq_n_s32(0);
                vecs_acc1.val[2] = vdupq_n_s32(0);
                vecs_acc1.val[3] = vdupq_n_s32(0);

                for(uint16_t k = 0; k < B_cols_div_8; k++)
                {
                    int8x8_t vec_a  =vld1_s8(A_data);

                    int8x8x4_t vecs_b;
                    vecs_b.val[0] = vld1_s8(B_data);
                    vecs_b.val[1] = vld1_s8(B_data + B_cols);
                    vecs_b.val[2] = vld1_s8(B_data + B_cols_mul_2);
                    vecs_b.val[3] = vld1_s8(B_data + B_cols_mul_3);

                    int16x8x4_t vecs_a_mul_b;
                    vecs_a_mul_b.val[0] = vmull_s8(vec_a, vecs_b.val[0]);
                    vecs_a_mul_b.val[1] = vmull_s8(vec_a, vecs_b.val[1]);
                    vecs_a_mul_b.val[2] = vmull_s8(vec_a, vecs_b.val[2]);
                    vecs_a_mul_b.val[3] = vmull_s8(vec_a, vecs_b.val[3]);

                    vecs_acc1.val[0] = vpadalq_s16(vecs_acc1.val[0], vecs_a_mul_b.val[0]);
                    vecs_acc1.val[1] = vpadalq_s16(vecs_acc1.val[1], vecs_a_mul_b.val[1]);
                    vecs_acc1.val[2] = vpadalq_s16(vecs_acc1.val[2], vecs_a_mul_b.val[2]);
                    vecs_acc1.val[3] = vpadalq_s16(vecs_acc1.val[3], vecs_a_mul_b.val[3]);

                    A_data += 8;
                    B_data += 8;
                }

                int64x2x4_t vecs_acc2;
                vecs_acc2.val[0] = vpaddlq_s32(vecs_acc1.val[0]);
                vecs_acc2.val[1] = vpaddlq_s32(vecs_acc1.val[1]);
                vecs_acc2.val[2] = vpaddlq_s32(vecs_acc1.val[2]);
                vecs_acc2.val[3] = vpaddlq_s32(vecs_acc1.val[3]);

                int64x2x2_t vecs_line_sum;
                vecs_line_sum.val[0] = vpaddq_s64(vecs_acc2.val[0], vecs_acc2.val[1]);
                vecs_line_sum.val[1] = vpaddq_s64(vecs_acc2.val[2], vecs_acc2.val[3]);

                vecs_line_sum.val[0] = vrshlq_s64(vecs_line_sum.val[0], vec_shift_bits_1);
                vecs_line_sum.val[1] = vrshlq_s64(vecs_line_sum.val[1], vec_shift_bits_1);

                vecs_line_sum.val[0] = vaddq_s64(vecs_line_sum.val[0], vec_bias);
                vecs_line_sum.val[1] = vaddq_s64(vecs_line_sum.val[1], vec_bias);

                vecs_line_sum.val[0] = vrshlq_s64(vecs_line_sum.val[0], vec_shift_bits_2);
                vecs_line_sum.val[1] = vrshlq_s64(vecs_line_sum.val[1], vec_shift_bits_2);

                int64_t result_0 = vgetq_lane_s64(vecs_line_sum.val[0], 0);
                int64_t result_1 = vgetq_lane_s64(vecs_line_sum.val[0], 1);
                int64_t result_2 = vgetq_lane_s64(vecs_line_sum.val[1], 0);
                int64_t result_3 = vgetq_lane_s64(vecs_line_sum.val[1], 1);

                *(C_data++) = (result_0 > 127) ? 127 : ((result_0 < -128) ? -128 : result_0);
                *(C_data++) = (result_1 > 127) ? 127 : ((result_1 < -128) ? -128 : result_1);
                *(C_data++) = (result_2 > 127) ? 127 : ((result_2 < -128) ? -128 : result_2);
                *(C_data++) = (result_3 > 127) ? 127 : ((result_3 < -128) ? -128 : result_3);
            }
        }
    }

    static void A_mul_Btranspose_pack2_even(Tensor *A, Tensor *B, Tensor *biases, Tensor *C)
    {
        uint16_t A_rows = A->dimension(1), A_cols = A->dimension(0);
        uint16_t B_rows = B->dimension(1), B_cols = B->dimension(0);

        assert(A_cols == B_cols);
        assert(biases->dimension(0) == A_rows);

        uint16_t B_cols_div_8 = B_cols / 8;
        int64x2_t vec_shift_bits_1 = vdupq_n_s64(-B->fractional_length());
        int64x2_t vec_shift_bits_2 = vdupq_n_s64(C->fractional_length() - A->fractional_length());

        int8_t *C_data = C->data();

        for(uint16_t i = 0; i < A_rows; i++)
        {
            int64x2_t vec_bias = vdupq_n_s64(*(biases->data() + i));
            int8_t *A_start = A->data() + i * A_cols;

            for(uint16_t j = 0; j < B_rows; j += 2)
            {
                int8_t *A_data = A_start;
                int8_t *B_data = B->data() + j * B_cols;

                int32x4x2_t vecs_acc1;
                vecs_acc1.val[0] = vdupq_n_s32(0);
                vecs_acc1.val[1] = vdupq_n_s32(0);

                for(uint16_t k = 0; k < B_cols_div_8; k++)
                {
                    int8x8_t vec_a = vld1_s8(A_data);

                    int8x8x2_t vecs_b;
                    vecs_b.val[0] = vld1_s8(B_data);
                    vecs_b.val[1] = vld1_s8(B_data + B_cols);

                    int16x8x2_t vecs_a_mul_b;
                    vecs_a_mul_b.val[0] = vmull_s8(vec_a, vecs_b.val[0]);
                    vecs_a_mul_b.val[1] = vmull_s8(vec_a, vecs_b.val[1]);

                    vecs_acc1.val[0] = vpadalq_s16(vecs_acc1.val[0], vecs_a_mul_b.val[0]);
                    vecs_acc1.val[1] = vpadalq_s16(vecs_acc1.val[1], vecs_a_mul_b.val[1]);

                    A_data += 8;
                    B_data += 8;
                }

                int64x2x2_t vecs_acc2;
                vecs_acc2.val[0] = vpaddlq_s32(vecs_acc1.val[0]);
                vecs_acc2.val[1] = vpaddlq_s32(vecs_acc1.val[1]);

                int64x2_t vec_line_sum = vpaddq_s64(vecs_acc2.val[0], vecs_acc2.val[1]);

                vec_line_sum = vrshlq_s64(vec_line_sum, vec_shift_bits_1);
                vec_line_sum = vaddq_s64(vec_line_sum, vec_bias);
                vec_line_sum = vrshlq_s64(vec_line_sum, vec_shift_bits_2);

                int64_t result_0 = vgetq_lane_s64(vec_line_sum, 0);
                int64_t result_1 = vgetq_lane_s64(vec_line_sum, 1);

                *(C_data++) = (result_0 > 127) ? 127 : ((result_0 < -128) ? -128 : result_0);
                *(C_data++) = (result_1 > 127) ? 127 : ((result_1 < -128) ? -128 : result_1);
            }
        }
    }

    static void A_mul_Btranspose_pack2_odd(Tensor *A, Tensor *B, Tensor *biases, Tensor *C)
    {
        uint16_t A_rows = A->dimension(1), A_cols = A->dimension(0);
        uint16_t B_rows = B->dimension(1), B_cols = B->dimension(0);
        uint16_t B_rows_min_1 = B_rows - 1;

        assert(A_cols == B_cols);
        assert(biases->dimension(0) == A_rows);

        uint16_t B_cols_div_8 = B_cols / 8;
        int8_t shift_bits_1 = -B->fractional_length();
        int8_t shift_bits_2 = A->fractional_length() - C->fractional_length();
        int64x2_t vec_shift_bits_1 = vdupq_n_s64(shift_bits_1);
        int64x2_t vec_shift_bits_2 = vdupq_n_s64(-shift_bits_2);
        int64_t roundoff = 1 << (shift_bits_2 - 1);

        int8_t *C_data = C->data();

        for(uint16_t i = 0; i < A_rows; i++)
        {
            int8_t bias = *(biases->data() + i);
            int64x2_t vec_bias = vdupq_n_s64(bias);
            int8_t *A_start = A->data() + i * A_cols;

            for(uint16_t j = 0; j < B_rows_min_1; j += 2)
            {
                int8_t *A_data = A_start;
                int8_t *B_data = B->data() + j * B_cols;

                int32x4x2_t vecs_acc1;
                vecs_acc1.val[0] = vdupq_n_s32(0);
                vecs_acc1.val[1] = vdupq_n_s32(0);

                for(uint16_t k = 0; k < B_cols_div_8; k++)
                {
                    int8x8_t vec_a = vld1_s8(A_data);

                    int8x8x2_t vecs_b;
                    vecs_b.val[0] = vld1_s8(B_data);
                    vecs_b.val[1] = vld1_s8(B_data + B_cols);

                    int16x8x2_t vecs_a_mul_b;
                    vecs_a_mul_b.val[0] = vmull_s8(vec_a, vecs_b.val[0]);
                    vecs_a_mul_b.val[1] = vmull_s8(vec_a, vecs_b.val[1]);

                    vecs_acc1.val[0] = vpadalq_s16(vecs_acc1.val[0], vecs_a_mul_b.val[0]);
                    vecs_acc1.val[1] = vpadalq_s16(vecs_acc1.val[1], vecs_a_mul_b.val[1]);

                    A_data += 8;
                    B_data += 8;
                }

                int64x2x2_t vecs_acc2;
                vecs_acc2.val[0] = vpaddlq_s32(vecs_acc1.val[0]);
                vecs_acc2.val[1] = vpaddlq_s32(vecs_acc1.val[1]);

                int64x2_t vec_line_sum = vpaddq_s64(vecs_acc2.val[0], vecs_acc2.val[1]);

                vec_line_sum = vrshlq_s64(vec_line_sum, vec_shift_bits_1);
                vec_line_sum = vaddq_s64(vec_line_sum, vec_bias);
                vec_line_sum = vrshlq_s64(vec_line_sum, vec_shift_bits_2);

                int64_t result_0 = vgetq_lane_s64(vec_line_sum, 0);
                int64_t result_1 = vgetq_lane_s64(vec_line_sum, 1);

                *(C_data++) = (result_0 > 127) ? 127 : ((result_0 < -128) ? -128 : result_0);
                *(C_data++) = (result_1 > 127) ? 127 : ((result_1 < -128) ? -128 : result_1);
            }

            int8_t *A_data = A_start;
            int8_t *B_data = B->data() + B_rows_min_1 * B_cols;

            int32x4_t vec_acc1 = vdupq_n_s32(0);

            for(uint16_t k = 0; k < B_cols_div_8; k++)
            {
                int8x8_t vec_a = vld1_s8(A_data);
                int8x8_t vec_b = vld1_s8(B_data);

                int16x8_t vec_a_mul_b = vmull_s8(vec_a, vec_b);

                vec_acc1 = vpadalq_s16(vec_acc1, vec_a_mul_b);

                A_data += 8;
                B_data += 8;
            }

            int64x2_t vec_acc2 = vpaddlq_s32(vec_acc1);

            int64_t line_sum = vgetq_lane_s64(vec_acc2, 0) + vgetq_lane_s64(vec_acc2, 1);

            line_sum = (((line_sum << shift_bits_1) + bias) + roundoff) >> shift_bits_2;

            *(C_data++) = (line_sum > 127) ? 127 : ((line_sum < -128) ? -128 : line_sum);
        }
    }

    static void A_mul_Btranspose(Tensor *A, Tensor *B, Tensor *biases, Tensor *C)
    {
        if(!(B->dimension(1) % 4))
            A_mul_Btranspose_pack4(A, B, biases, C);
        else if(!(B->dimension(1) % 2))
            A_mul_Btranspose_pack2_even(A, B, biases, C);
        else
            A_mul_Btranspose_pack2_odd(A, B, biases, C);
    }

    static void shift_and_copy(Tensor *input_tensor, int8_t *output_data, int8_t shift_bits)
    {
        size_t total_size_div_16 = input_tensor->total_size() / 16;
        uint8_t total_size_mod_16 = input_tensor->total_size() % 16;

        int8x16_t vec_shift_bits = vdupq_n_s8(shift_bits);

        int8_t *input_data = input_tensor->data();

        for(size_t i = 0; i < total_size_div_16; i++)
        {
            int8x16_t vec_in = vld1q_s8(input_data);
            int8x16_t vec_out = vrshlq_s8(vec_in, vec_shift_bits);
            vst1q_s8(output_data, vec_out);

            input_data += 16;
            output_data += 16;
        }

        int8x16_t vec_in = vdupq_n_s8(0);

        for(uint8_t i = 0; i < total_size_mod_16; i++)
            vec_in = vld1q_lane_s8(input_data++, vec_in, i);

        int8x16_t vec_out = vrshlq_s8(vec_in, vec_shift_bits);

        for(uint8_t i = 0; i < total_size_mod_16; i++)
            vst1q_lane_s8(output_data++, vec_out, i);
    }

};

#endif /* __CI4A_TENSORMATH_H */
