#ifndef __CI4A_INPUTLAYER_H
#define __CI4A_INPUTLAYER_H

#include <iostream>
#include <string>
#include <vector>
#include <cassert>

#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <opencv2/opencv.hpp>

#include "tensor/Tensor.h"

using namespace std;

class InputLayer
{

private:

    uint16_t _width, _height, _channel;

    string _image_filename;
    string _mean_filename;

    Tensor *_output_tensor;

    cv::Mat _mean;

    vector<cv::Mat> _channels;

public:

    /* Constructor */
    InputLayer(uint16_t width, uint16_t height, uint16_t channel, string mean_filename)
    {
        /* Members initialization */
        _width = width;
        _height = height;
        _channel = channel;
        _mean_filename = mean_filename;

        /* Read mean data from file */
        caffe::BlobProto blob_proto;
        caffe::ReadProtoFromBinaryFileOrDie(_mean_filename.c_str(), &blob_proto);
        caffe::Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);

        /* Convert mean_blob to a opencv mat */
        vector<cv::Mat> channels;
        float *data = mean_blob.mutable_cpu_data();
        for(uint16_t i = 0; i < _channel; i++)
        {
            cv::Mat ch(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(ch);
            data += mean_blob.height() * mean_blob.width();
        }
        cv::Mat mean_orig;
        cv::merge(channels, mean_orig);

        /* Calculate mean value on each image channel */
        cv::Scalar channel_mean = cv::mean(mean_orig);
        _mean = cv::Mat(_height, _width, CV_16SC3, channel_mean);

        /* Create the output tensor object */
        _output_tensor = new Tensor(_width, _height, _channel, 0);
        _output_tensor->allocate();

        /* Create a vector of opencv mat which shares the memory with _output_tensor */
        for(uint16_t ch = 0; ch < _channel; ch++)
        {
            cv::Mat temp(_height, _width, CV_8SC1, _output_tensor->data() + ch * _height * _width);
            _channels.push_back(temp);
        }
    }

    /* Set the input image to be read */
    void set_image_filename(string image_filename)
    {
        _image_filename = image_filename;
    }

    /* Implementation of forward propagation */
    void forward(void)
    {
        cv::Size input_geometry(_width, _height);

        /* Read the image and resize if necessary */
        cv::Mat image = cv::imread(_image_filename.c_str());
        if(image.size() != input_geometry)
            cv::resize(image, image, input_geometry);

        /* Mean subtraction */
        image.convertTo(image, CV_16SC3);
        cv::subtract(image, _mean, image);
        image.convertTo(image, CV_8SC3);

        /* Load the image data into _outpu_tensor */
        cv::split(image, _channels);
    }

    /* Return a pointer to the output tensor */
    Tensor *ptr_output_tensor(void)
    {
        return _output_tensor;
    }
};

#endif /* __CI4A_INPUTLAYER_H */
