#include <stdint.h>
#include <math.h>
#include <stddef.h>


int symmetric_dequantize(float *dequantized_data)
{
    for (unsigned int i = 0; i < data_size; i++)
    {
        dequantized_data[i] = (float)quantized_data[i] * scale;
    }

    return 0;
}


int asymmetric_dequantize(float *dequantized_data)
{
    for (unsigned int i = 0; i < data_size; i++)
    {
        dequantized_data[i] = ((float)quantized_data[i] - zero_point) * scale;
    }

    return 0;
}