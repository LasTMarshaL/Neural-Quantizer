#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "../Algorithms/Dequantizer.h"


void test_symmetric_basic(void)
{
    float dequantized_data[3];

    int status = symmetric_dequantizer((int8_t[]){10, 20, 30}, dequantized_data, 3, 0.1f);

    assert(status == DEQUANTIZATION_SUCCESS);

    assert(fabsf(dequantized_data[0] - 1.0f) < 0.000001f);
    assert(fabsf(dequantized_data[1] - 2.0f) < 0.000001f);
    assert(fabsf(dequantized_data[2] - 3.0f) < 0.000001f);
}

void test_symmetric_null_quantized_pointer(void)
{
    int status = symmetric_dequantizer(NULL, (float[5]){0}, 5, 0.1f);

    assert(status == DEQUANTIZATION_NULL_POINTER);
}


void test_symmetric_null_dequantized_pointer(void)
{
    int status = symmetric_dequantizer((int8_t[]){12, 45, 67}, NULL, 1, 0.1f);

    assert(status == DEQUANTIZATION_NULL_POINTER);
}


void test_symmetric_zero_size(void)
{  
    int status = symmetric_dequantizer((int8_t[]){34, 67, 45, -37}, (float[4]){0}, 0, 0.1f);

    assert(status == DEQUANTIZATION_INVALID_SIZE);
}


void test_symmetric_negative_scale(void)
{
    int status = symmetric_dequantizer((int8_t[]){34, 67, 45, -37}, (float[4]){0}, 4, -0.5f);

    assert(status == DEQUANTIZATION_INVALID_SCALE);
}


void test_symmetric_infinite_scale(void)
{
    int status = symmetric_dequantizer((int8_t[]){34, 67, 45, -37}, (float[4]){0}, 4,  INFINITY);

    assert(status == DEQUANTIZATION_INVALID_SCALE);
}


void test_asymmetric_basic(void)
{
    float dequantized_data[3];

    int status = asymmetric_dequantizer((int8_t[]){10, 20, 30}, dequantized_data, 3, 0.1f, 5);

    assert(status == DEQUANTIZATION_SUCCESS);

    assert(fabsf(dequantized_data[0] - 0.5f) < 0.000001f);
    assert(fabsf(dequantized_data[1] - 1.5f) < 0.000001f);
    assert(fabsf(dequantized_data[2] - 2.5f) < 0.000001f);
}


void test_asymmetric_null_quantized_pointer(void)
{
    int status = asymmetric_dequantizer(NULL, (float[5]){5}, 1, 0.1f, 7);

    assert(status == DEQUANTIZATION_NULL_POINTER);
}


void test_asymmetric_null_dequantized_pointer(void)
{
    int status = asymmetric_dequantizer((int8_t[]){12, 45, 67}, NULL, 1, 0.1f, 15);

    assert(status == DEQUANTIZATION_NULL_POINTER);
}


void test_asymmetric_zero_size(void)
{
    int status = asymmetric_dequantizer((int8_t[]){34, 67, 45, -37}, (float[4]){0}, 0, 0.1f, -9);

    assert(status == DEQUANTIZATION_INVALID_SIZE);
}


void test_asymmetric_negative_scale(void)
{
    int status = asymmetric_dequantizer((int8_t[]){34, 67, 45, -37}, (float[4]){0}, 4, -0.5f, -8);

    assert(status == DEQUANTIZATION_INVALID_SCALE);
}


void test_asymmetric_infinite_scale(void)
{
    int status = asymmetric_dequantizer((int8_t[]){34, 67, 45, -37}, (float[4]){0}, 4, INFINITY, 9);

    assert(status == DEQUANTIZATION_INVALID_SCALE);
}


int main(void)
{
    printf("Tests are launched...\n");

    test_symmetric_basic(); printf("1. symmetric_basic OK\n");
    test_symmetric_null_quantized_pointer(); printf("2. symmetric_null_quantized OK\n");
    test_symmetric_null_dequantized_pointer(); printf("3. symmetric_null_dequantized OK\n");
    test_symmetric_zero_size(); printf("4. symmetric_negative_size OK\n");
    test_symmetric_negative_scale(); printf("5. symmetric_negative_scale OK\n");
    test_symmetric_infinite_scale(); printf("6. symmetric_infinite_scale OK\n");

    test_asymmetric_basic(); printf("7. asymmetric_basic OK\n");
    test_asymmetric_null_quantized_pointer(); printf("8. asymmetric_null_quantized OK\n");
    test_asymmetric_null_dequantized_pointer(); printf("9. asymmetric_null_dequantized OK\n");
    test_asymmetric_zero_size(); printf("10. asymmetric_negative_size OK\n");
    test_asymmetric_negative_scale(); printf("11. asymmetric_negative_scale OK\n");
    test_asymmetric_infinite_scale(); printf("12. asymmetric_infinite_scale OK\n");

    printf("All tests finished!\n");
    return 0;
}