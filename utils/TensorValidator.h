#ifndef __CI4A_TENSORVALIDATOR_H
#define __CI4A_TENSORVALIDATOR_H

#include <iostream>
#include <fstream>
#include <cassert>

#include "tensor/Tensor.h"

using namespace std;

class TensorValidator
{

private:

public:

    static uint32_t compare_with_file(Tensor *tensorUnderTest, string goldFilename, uint32_t length)
    {
        /* Open file */
        ifstream ifile(goldFilename.c_str(), ios::in | ios::binary);
        assert(ifile.is_open());

        /* Acquire the number of bytes store in this file */
        ifile.seekg(0, ios::end);
        size_t fileSize = (size_t)ifile.tellg() + 1;

        /* Read all data from file */
        float *readBuff = new float[fileSize / 4];
        ifile.seekg(0, ios::beg);
        ifile.read((char *)readBuff, fileSize);
        ifile.close();

        /* Variable to count the number of different bytes */
        uint32_t err = 0;

        /* Compare the tensor with "golden" refenence */
        for(uint32_t i = 0; i < length; i++)
        {
            int8_t dut = *(tensorUnderTest->data() + i);
            int8_t gold = (int8_t)(readBuff[i] * pow(2, tensorUnderTest->fractional_length()));

            if(dut != gold)
            {
                cout << i  << ":\t" << (int32_t)gold << '\t' << (int32_t)dut << endl;
                err++;
            }
        }

        cout << "error: " << err << endl;

        delete[] readBuff;

        return err;
    }

};

#endif /* __CI4A_TENSORVALIDATOR */
