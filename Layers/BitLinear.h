#pragma once
#include <ap_int.h>
#include "VecAdd.h"

// Bit‐linear matmul + linear:
//
//   weights ∈ {+1,−1}, packed as ap_uint<1>
//   bit = 1 → +1, bit = 0 → −1

template<typename T, int ROWS, int HIDDEN, int COLS>
void bitlinear_matmul(
    T            A[ROWS][HIDDEN],
    ap_uint<1>   W[HIDDEN][COLS],
    T            C[ROWS][COLS]
) {
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=W complete dim=2
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
#pragma HLS PIPELINE II=1
            T sum = 0;
            // fully unroll if HIDDEN small; else use factor=n
#pragma HLS UNROLL
            for (int k = 0; k < HIDDEN; k++) {
                // 1→+A, 0→−A
                sum += W[k][j] ?  A[i][k]
                                : -A[i][k];
            }
            C[i][j] = sum;
        }
    }
}

template<typename T, int ROWS, int HIDDEN, int COLS>
void bitlinear(
    T            input[ROWS][HIDDEN],
    ap_uint<1>   weights[HIDDEN][COLS],
    T            biases[COLS],
    T            result[ROWS][COLS]
) {
    T tmp[ROWS][COLS];
#pragma HLS ARRAY_PARTITION variable=tmp block factor=8 dim=2
    // matmul
    bitlinear_matmul<T,ROWS,HIDDEN,COLS>(input, weights, tmp);
    // bias add
    for (int i = 0; i < ROWS; i++) {
#pragma HLS PIPELINE II=1
        vecadd<T,COLS>(tmp[i], biases, result[i]);
    }
}

//
// Ternary‐linear matmul + linear:
//
//   weights ∈ {+1,0,−1}, stored as signed 2‐bit (ap_int<2>)
//     code: +1 →  1;  0 →  0;  −1 → −1
//
template<typename T, int ROWS, int HIDDEN, int COLS>
void ternary_matmul(
    T           A[ROWS][HIDDEN],
    ap_int<2>   W[HIDDEN][COLS],
    T           C[ROWS][COLS]
) {
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=W complete dim=2
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
#pragma HLS PIPELINE II=1
            T sum = 0;
#pragma HLS UNROLL
            for (int k = 0; k < HIDDEN; k++) {
                ap_int<2> w = W[k][j];
                if      (w ==  1) sum += A[i][k];
                else if (w == -1) sum -= A[i][k];
                // w==0 → no operation
            }
            C[i][j] = sum;
        }
    }
}

template<typename T, int ROWS, int HIDDEN, int COLS>
void ternary_linear(
    T           input[ROWS][HIDDEN],
    ap_int<2>   weights[HIDDEN][COLS],
    T           biases[COLS],
    T           result[ROWS][COLS]
) {
    T tmp[ROWS][COLS];
#pragma HLS ARRAY_PARTITION variable=tmp block factor=8 dim=2
    // matmul
    ternary_matmul<T,ROWS,HIDDEN,COLS>(input, weights, tmp);
    // bias add
    for (int i = 0; i < ROWS; i++) {
#pragma HLS PIPELINE II=1
        vecadd<T,COLS>(tmp[i], biases, result[i]);
    }
}
