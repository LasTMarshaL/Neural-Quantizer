#ifndef DEQUANTIZER_H
#define DEQUANTIZER_H


#include <stdint.h>


#define DEQUANTIZATION_SUCCESS 0
#define DEQUANTIZATION_NULL_POINTER -1
#define DEQUANTIZATION_INVALID_SIZE -2
#define DEQUANTIZATION_INVALID_SCALE -3


int symmetric_dequantizer(const int8_t *quantized_data, float *dequantized_data, unsigned int size, float scale);
int asymmetric_dequantizer(const int8_t *quantized_data, float *dequantized_data, unsigned int size, float scale, int8_t zero_point);


#endif