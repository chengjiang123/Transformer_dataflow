#pragma once
#include "VecAdd.h"
#include "MatMul.h"

template<typename T, int rows, int hidden, int cols>
void linear(
	T input[rows][hidden],
	T weights[hidden][cols], 
	T biases[cols],
	T result[rows][cols]
	) {
	T tmp[rows][cols];
	matmul<T, rows, hidden, cols>(input, weights, tmp);
	for (int i = 0; i < rows; i++) {
		vecadd<T, cols>(tmp[i], biases, result[i]);
	}
}

template<typename T, int rows, int hidden, int cols>
void bitlinear(
    T input[rows][hidden],          // Quantized input (e.g., int8)
    int8_t weights[hidden][cols],   // Ternary weights: -1, 0, 1
    T beta[cols],                   // scaling factors
    T biases[cols],                 
    T result[rows][cols]           
) {
    T scaled_result[rows][cols];
    #pragma HLS ARRAY_PARTITION variable=scaled_result cyclic factor=8 dim=2
    
    // Ternary matrix multiplication
    ternary_matmul<T, rows, hidden, cols>(input, weights, scaled_result);
    
bitlinear_scale_row:
    for (int i = 0; i < rows; i++) {
    bitlinear_scale_col:
        for (int j = 0; j < cols; j++) {
            #pragma HLS UNROLL factor=4
            scaled_result[i][j] = scaled_result[i][j] * beta[j];
        }
    }
    
    // Add bias, CJ: do we need this?
bitlinear_bias_row:
    for (int i = 0; i < rows; i++) {
        vecadd<T, cols>(scaled_result[i], biases, result[i]);
    }
}
