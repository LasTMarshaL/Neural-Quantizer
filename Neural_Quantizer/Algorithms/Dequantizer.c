#include <stdint.h>
#include <math.h>
#include <stddef.h>


#define DEQUANTIZATION_SUCCESS 0
#define DEQUANTIZATION_NULL_POINTER -1
#define DEQUANTIZATION_INVALID_SIZE -2
#define DEQUANTIZATION_INVALID_SCALE -3


int symmetric_dequantizer(const int8_t *quantized_data, float *dequantized_data, unsigned int size, float scale)
{
    if (quantized_data == NULL || dequantized_data == NULL)
    {
        return DEQUANTIZATION_NULL_POINTER;
    }

    if (size == 0)
    {
        return DEQUANTIZATION_INVALID_SIZE;
    }

    if (!isfinite(scale) || scale <= 0.0f)
    {
        return DEQUANTIZATION_INVALID_SCALE;
    }

    for (unsigned int i = 0; i < size; i++)
    {
        dequantized_data[i] = (float)quantized_data[i] * scale;
    }

    return DEQUANTIZATION_SUCCESS;
}


int asymmetric_dequantizer(const int8_t *quantized_data, float *dequantized_data, unsigned int size, float scale, int8_t zero_point)
{
    if (quantized_data == NULL || dequantized_data == NULL)
    {
        return DEQUANTIZATION_NULL_POINTER;
    }

    if (size == 0)
    {
        return DEQUANTIZATION_INVALID_SIZE;
    }

    if (!isfinite(scale) || scale <= 0.0f)
    {
        return DEQUANTIZATION_INVALID_SCALE;
    }

    for (unsigned int i = 0; i < size; i++)
    {
        dequantized_data[i] = ((float)quantized_data[i] - zero_point) * scale;
    }

    return DEQUANTIZATION_SUCCESS;
}