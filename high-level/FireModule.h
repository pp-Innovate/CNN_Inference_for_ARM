#include <iostream>
#include <cassert>
#include <string>
#include <unistd.h>
#include <sys/time.h>

#include "tensor/Tensor.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/ReLULayer.h"
#include "layers/ConcatLayer.h"
#include "utils/ParamLoader.h"

using namespace std;

class FireModule_type1
{

private:

    ConvolutionalLayer *_squeeze1x1;
    ReLULayer *_relu_squeeze1x1;
    ConvolutionalLayer *_expand1x1;
    ReLULayer *_relu_expand1x1;
    ConvolutionalLayer *_expand3x3;
    ReLULayer *_relu_expand3x3;
    ConcatLayer *_concat;

public:

    FireModule_type1(
            string name, 
            ParamLoader *param_loader, 
            Tensor *input_tensor, 
            uint16_t squeeze_channel_out, uint16_t expand_channel_out, 
            int8_t squeeze_fl_param, int8_t squeeze_fl_out, 
            int8_t expand1x1_fl_param, int8_t expand1x1_fl_out, 
            int8_t expand3x3_fl_param, int8_t expand3x3_fl_out, 
            int8_t concat_fl_out 
        )
    {
        _squeeze1x1 = new ConvolutionalLayer(input_tensor, 1, squeeze_channel_out, 1, 0, squeeze_fl_param, squeeze_fl_out);
        param_loader->load(name + "/squeeze1x1", "weight", _squeeze1x1->ptr_weights_tensor());
        param_loader->load(name + "/squeeze1x1", "bias", _squeeze1x1->ptr_biases_tensor());
        _relu_squeeze1x1 = new ReLULayer(_squeeze1x1->ptr_output_tensor());

        _expand1x1 = new ConvolutionalLayer(_relu_squeeze1x1->ptr_output_tensor(), 1, expand_channel_out, 1, 0, expand1x1_fl_param, expand1x1_fl_out);
        param_loader->load(name + "/expand1x1", "weight", _expand1x1->ptr_weights_tensor());
        param_loader->load(name + "/expand1x1", "bias", _expand1x1->ptr_biases_tensor());
        _relu_expand1x1 = new ReLULayer(_expand1x1->ptr_output_tensor());

        _expand3x3 = new ConvolutionalLayer(_relu_squeeze1x1->ptr_output_tensor(), 3, expand_channel_out, 1, 1, expand3x3_fl_param, expand3x3_fl_out);
        param_loader->load(name + "/expand3x3", "weight", _expand3x3->ptr_weights_tensor());
        param_loader->load(name + "/expand3x3", "bias", _expand3x3->ptr_biases_tensor());
        _relu_expand3x3 = new ReLULayer(_expand3x3->ptr_output_tensor());

        _concat = new ConcatLayer(_relu_expand1x1->ptr_output_tensor(), _relu_expand3x3->ptr_output_tensor(), concat_fl_out);
    }

    void forward(void)
    {
        _squeeze1x1->forward();
        _relu_squeeze1x1->forward();

        _expand1x1->forward();
        _relu_expand1x1->forward();

        _expand3x3->forward();
        _relu_expand3x3->forward();

        _concat->forward();
    }

    Tensor *ptr_squeeze1x1_output_tensor(void)
    {
        return _relu_squeeze1x1->ptr_output_tensor();
    }

    Tensor *ptr_expand1x1_output_tensor(void)
    {
        return _relu_expand1x1->ptr_output_tensor();
    }

    Tensor *ptr_expand3x3_output_tensor(void)
    {
        return _relu_expand3x3->ptr_output_tensor();
    }

    Tensor *ptr_output_tensor(void)
    {
        return _concat->ptr_output_tensor();
    }

};

class FireModule_type2
{

private:

    ConvolutionalLayer *_squeeze3x3;
    ReLULayer *_relu_squeeze3x3;
    ConvolutionalLayer *_expand1x1;
    ReLULayer *_relu_expand1x1;
    ConvolutionalLayer *_expand3x3;
    ReLULayer *_relu_expand3x3;
    ConcatLayer *_concat;

public:

    FireModule_type2(
            string name, 
            ParamLoader *param_loader, 
            Tensor *input_tensor, 
            uint16_t squeeze_channel_out, uint16_t expand_channel_out, 
            int8_t squeeze_fl_param, int8_t squeeze_fl_out, 
            int8_t expand1x1_fl_param, int8_t expand1x1_fl_out, 
            int8_t expand3x3_fl_param, int8_t expand3x3_fl_out, 
            int8_t concat_fl_out 
        )
    {
        _squeeze3x3 = new ConvolutionalLayer(input_tensor, 3, squeeze_channel_out, 2, 1, squeeze_fl_param, squeeze_fl_out);
        param_loader->load(name + "/squeeze3x3", "weight", _squeeze3x3->ptr_weights_tensor());
        param_loader->load(name + "/squeeze3x3", "bias", _squeeze3x3->ptr_biases_tensor());
        _relu_squeeze3x3 = new ReLULayer(_squeeze3x3->ptr_output_tensor());

        _expand1x1 = new ConvolutionalLayer(_relu_squeeze3x3->ptr_output_tensor(), 1, expand_channel_out, 1, 0, expand1x1_fl_param, expand1x1_fl_out);
        param_loader->load(name + "/expand1x1", "weight", _expand1x1->ptr_weights_tensor());
        param_loader->load(name + "/expand1x1", "bias", _expand1x1->ptr_biases_tensor());
        _relu_expand1x1 = new ReLULayer(_expand1x1->ptr_output_tensor());

        _expand3x3 = new ConvolutionalLayer(_relu_squeeze3x3->ptr_output_tensor(), 3, expand_channel_out, 1, 1, expand3x3_fl_param, expand3x3_fl_out);
        param_loader->load(name + "/expand3x3", "weight", _expand3x3->ptr_weights_tensor());
        param_loader->load(name + "/expand3x3", "bias", _expand3x3->ptr_biases_tensor());
        _relu_expand3x3 = new ReLULayer(_expand3x3->ptr_output_tensor());

        _concat = new ConcatLayer(_relu_expand1x1->ptr_output_tensor(), _relu_expand3x3->ptr_output_tensor(), concat_fl_out);
    }

    void forward(void)
    {
        _squeeze3x3->forward();
        _relu_squeeze3x3->forward();

        _expand1x1->forward();
        _relu_expand1x1->forward();

        _expand3x3->forward();
        _relu_expand3x3->forward();

        _concat->forward();
    }

    Tensor *ptr_squeeze3x3_output_tensor(void)
    {
        return _relu_squeeze3x3->ptr_output_tensor();
    }

    Tensor *ptr_expand1x1_output_tensor(void)
    {
        return _relu_expand1x1->ptr_output_tensor();
    }

    Tensor *ptr_expand3x3_output_tensor(void)
    {
        return _relu_expand3x3->ptr_output_tensor();
    }

    Tensor *ptr_output_tensor(void)
    {
        return _concat->ptr_output_tensor();
    }
};
