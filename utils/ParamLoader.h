#ifndef __CI4A_PARAMLOADER_H
#define __CI4A_PARAMLOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <string>
#include <cstring>
#include <cassert>

#include "tensor/Tensor.h"

using namespace std;

typedef struct
{
    char layerName[128];
    char paramType[32];
    size_t start;
    size_t length;
    int32_t fractionalLength;
}ParamDescriptor;

class ParamLoader
{

private:

    string _description_filename;
    string _param_filename;

    vector<ParamDescriptor> _param_descriptors;

public:

    /* Constructor */
    ParamLoader(string description_filename, string param_filename)
    {
        /* Member variables initialization */
        _description_filename = description_filename;
        _param_filename = param_filename;

        /* Open description file */
        ifstream ifile(_description_filename.c_str(), ios::in);
        assert(ifile.is_open());

        /* Read the first line of description file */
        string line_buffer;
        getline(ifile, line_buffer);

        /* Parse the description content */
        while(getline(ifile, line_buffer))
        {
            ParamDescriptor descriptor;
            sscanf(
                    line_buffer.c_str(), "%s\t\t%s\t\t%ld\t\t%ld\t\t%d\n",
                    descriptor.layerName, descriptor.paramType,
                    &descriptor.start, &descriptor.length,
                    &descriptor.fractionalLength
            );

            _param_descriptors.push_back(descriptor);
        }

        /* Close file */
        ifile.close();
    }

    /* Load parameters from file */
    void load(string layer_name, string param_type, Tensor *tensor)
    {
        /* Search for the descriptor of desired parameters */
        uint16_t index;
        for(index = 0; index < _param_descriptors.size(); index++)
            if(!strcmp(_param_descriptors[index].layerName, layer_name.c_str()) && !strcmp(_param_descriptors[index].paramType, param_type.c_str()))
                break;

        assert(index < _param_descriptors.size());
        assert(tensor->total_size() == _param_descriptors[index].length);
        assert(tensor->fractional_length() == _param_descriptors[index].fractionalLength);

        /* Open parameter file */
        ifstream ifile(_param_filename.c_str(), ios::in | ios::binary);
        assert(ifile.is_open());

        /* Read desired parameters */
        ifile.seekg(_param_descriptors[index].start, ios::beg);
        ifile.read((char *)tensor->data(), _param_descriptors[index].length);

        /* Close file */
        ifile.close();
    }

};

#endif /* __CI4A_PARAMLOADER_H */ 
