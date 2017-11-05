#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include <sys/time.h>
#include <cassert>

#include <cmdline.h>

#include "CI4A.h"

using namespace std;

void find_top_n(Tensor *input_tensor, uint8_t n, uint16_t *top_n_id);

const string ZynqNet_Root = "../example/ZynqNet/";

int main(int argc, char *argv[])
{
    cmdline::parser argParser;
    argParser.parse_check(argc, argv);

    ParamLoader param_loader(
            ZynqNet_Root + "data/ZynqNet_Quantized_description.txt", 
            ZynqNet_Root + "data/ZynqNet_Quantized_8bit.dat"
    );

    InputLayer data(256, 256, 3, ZynqNet_Root + "imagenet_mean.binaryproto");
    data.set_image_filename(ZynqNet_Root + "test.JPEG");

    ConvolutionalLayer conv1(data.ptr_output_tensor(), 3, 64, 2, 1, 7, -3);
    param_loader.load("conv1", "weight", conv1.ptr_weights_tensor());
    param_loader.load("conv1", "bias", conv1.ptr_biases_tensor());
    ReLULayer relu_conv1(conv1.ptr_output_tensor());

    FireModule_type2 fire2("fire2", &param_loader, relu_conv1.ptr_output_tensor(), 16, 64, 6, -5, 6, -5, 7, -5, -5);
    FireModule_type1 fire3("fire3", &param_loader, fire2.ptr_output_tensor(), 16, 64, 6, -6, 7, -5, 7, -7, -7);
    FireModule_type2 fire4("fire4", &param_loader, fire3.ptr_output_tensor(), 32, 128, 8, -7, 7, -6, 8, -7, -7);
    FireModule_type1 fire5("fire5", &param_loader, fire4.ptr_output_tensor(), 32, 128, 7, -7, 7, -6, 7, -7, -7);
    FireModule_type2 fire6("fire6", &param_loader, fire5.ptr_output_tensor(), 64, 256, 8, -7, 7, -6, 8, -7, -7);
    FireModule_type1 fire7("fire7", &param_loader, fire6.ptr_output_tensor(), 64, 192, 7, -6, 8, -5, 8, -6, -6);
    FireModule_type2 fire8("fire8", &param_loader, fire7.ptr_output_tensor(), 112, 256, 8, -5, 8, -4, 8, -5, -5);
    FireModule_type1 fire9("fire9", &param_loader, fire8.ptr_output_tensor(), 112, 368, 8, -4, 8, -1, 9, -2, -3);

    ConvolutionalLayer conv10_split1(fire9.ptr_output_tensor(), 1, 512, 1, 0, 10, -1);
    param_loader.load("conv10/split1", "weight", conv10_split1.ptr_weights_tensor());
    param_loader.load("conv10/split1", "bias", conv10_split1.ptr_biases_tensor());

    ConvolutionalLayer conv10_split2(fire9.ptr_output_tensor(), 1, 512, 1, 0, 10, 0);
    param_loader.load("conv10/split2", "weight", conv10_split2.ptr_weights_tensor());
    param_loader.load("conv10/split2", "bias", conv10_split2.ptr_biases_tensor());

    ConcatLayer conv10(conv10_split1.ptr_output_tensor(), conv10_split2.ptr_output_tensor(), 0);

    GlobalPoolingLayer pool10(conv10.ptr_output_tensor());

    data.forward();
    conv1.forward();
    relu_conv1.forward();
    fire2.forward();
    fire3.forward();
    fire4.forward();
    fire5.forward();
    fire6.forward();
    fire7.forward();
    fire8.forward();
    fire9.forward();
    conv10_split1.forward();
    conv10_split2.forward();
    conv10.forward();
    pool10.forward();

    //ifstream ifile("/home/firefly/Documents/caffe-1.0/data/ilsvrc12/val.txt", ios::in);

    //string sample;
    //string file_name;
    //int16_t label;
    //uint16_t top_5[5];
    //int32_t top_1_hit = 0;
    //int32_t top_5_hit = 0;
    //int32_t total_cnt = 0;

    //while(getline(ifile, sample))
    //{
    //    int space_pos = sample.find(" ", 0);
    //    file_name = sample.substr(0, space_pos);
    //    label = atoi(sample.substr(space_pos + 1).c_str());

    //    data.set_image_filename("/mnt/" + file_name);
    //    data.forward();
    //    conv1.forward();
    //    relu_conv1.forward();
    //    fire2.forward();
    //    fire3.forward();
    //    fire4.forward();
    //    fire5.forward();
    //    fire6.forward();
    //    fire7.forward();
    //    fire8.forward();
    //    fire9.forward();
    //    conv10_split1.forward();
    //    conv10_split2.forward();
    //    conv10.forward();
    //    pool10.forward();

    //    find_top_n(pool10.ptr_output_tensor(), 5, top_5);

    //    for(uint8_t i = 0; i < 5; i++)
    //    {
    //        if(top_5[i] == label)
    //        {
    //            top_5_hit++;

    //            if(i == 0)
    //                top_1_hit++;

    //            break;
    //        }
    //    }

    //    total_cnt ++;

    //    if(!(total_cnt % 1000))
    //        cout << total_cnt << '\t' << top_1_hit/(float)total_cnt << '\t' << top_5_hit/(float)total_cnt << endl;
    //}

    //cout << "top-1: " << top_1_hit/(float)total_cnt << '\t' << "top-5: " << top_5_hit/(float)total_cnt << endl;

    //ifile.close();

    //TensorValidator::compare_with_file(relu_conv1.ptr_output_tensor(), ZynqNet_Root + "data/conv1.dat", relu_conv1.ptr_output_tensor()->total_size());
    //TensorValidator::compare_with_file(pool10.ptr_output_tensor(), ZynqNet_Root + "data/pool10.dat", pool10.ptr_output_tensor()->total_size());
    //
    timeval begin, after;
    gettimeofday(&begin, NULL);
    for(uint16_t i = 0; i < 100; i++)
    {
        data.forward();
        conv1.forward();
        relu_conv1.forward();
        fire2.forward();
        fire3.forward();
        fire4.forward();
        fire5.forward();
        fire6.forward();
        fire7.forward();
        fire8.forward();
        fire9.forward();
        conv10_split1.forward();
        conv10_split2.forward();
        conv10.forward();
        pool10.forward();
    }
    gettimeofday(&after, NULL);
    uint32_t time_ms = (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000;
    cout << time_ms << "ms" << endl;
    cout << 1000 * 100 / (float)time_ms << "fps" << endl;

    gettimeofday(&begin, NULL);
    data.forward();
    gettimeofday(&after, NULL);
    cout << "data " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;

    gettimeofday(&begin, NULL);
    conv1.forward();
    relu_conv1.forward();
    gettimeofday(&after, NULL);
    cout << "conv1 " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;

    gettimeofday(&begin, NULL);
    fire2.forward();
    gettimeofday(&after, NULL);
    cout << "fire2 " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;

    gettimeofday(&begin, NULL);
    fire3.forward();
    gettimeofday(&after, NULL);
    cout << "fire3 " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;

    gettimeofday(&begin, NULL);
    fire4.forward();
    gettimeofday(&after, NULL);
    cout << "fire4 " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;

    gettimeofday(&begin, NULL);
    fire5.forward();
    gettimeofday(&after, NULL);
    cout << "fire5 " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;

    gettimeofday(&begin, NULL);
    fire6.forward();
    gettimeofday(&after, NULL);
    cout << "fire6 " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;

    gettimeofday(&begin, NULL);
    fire7.forward();
    gettimeofday(&after, NULL);
    cout << "fire7 " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;

    gettimeofday(&begin, NULL);
    fire8.forward();
    gettimeofday(&after, NULL);
    cout << "fire8 " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;

    gettimeofday(&begin, NULL);
    fire9.forward();
    gettimeofday(&after, NULL);
    cout << "fire9 " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;

    gettimeofday(&begin, NULL);
    conv10_split1.forward();
    conv10_split2.forward();
    conv10.forward();
    gettimeofday(&after, NULL);
    cout << "conv10 " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;
    
    gettimeofday(&begin, NULL);
    pool10.forward();
    gettimeofday(&after, NULL);
    cout << "pool10 " << (after.tv_sec - begin.tv_sec) * 1000 + (after.tv_usec - begin.tv_usec) / 1000 << "ms" << endl;

    return 0;
}

void find_top_n(Tensor *input_tensor, uint8_t n, uint16_t *top_n_id)
{
    uint16_t *index_table = new uint16_t[input_tensor->total_size()];

    for(uint16_t i = 0; i < input_tensor->total_size(); i++)
        index_table[i] = i;

    for(uint8_t i = 0; i < n; i++)
    {
        uint16_t max_id = i;

        for(uint16_t j = i; j < input_tensor->total_size(); j++)
            if(input_tensor->data()[index_table[j]] > input_tensor->data()[index_table[max_id]])
                max_id = j;

        uint16_t temp = index_table[i];
        index_table[i] = index_table[max_id];
        index_table[max_id] = temp;

        top_n_id[i] = index_table[i];
    }

    delete[] index_table;
}
