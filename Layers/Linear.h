#pragma once
#include <cmath>
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





template<typename T>
static inline int round_sym(T x) {
  // symmetric round-to-nearest (handles negatives properly)
  return (x >= (T)0) ? (int)std::floor(x + (T)0.5)
                     : (int)std::ceil (x - (T)0.5);
}

template<typename T, int rows, int hidden, int cols, int B = 8>
void bitlinear_binary(
    T input[rows][hidden],
    T weights[hidden][cols],
    T biases[cols],
    T result[rows][cols],
    T eps = (T)1e-8
){
  T sum_w = (T)0;
  T sum_abs = (T)0;
  const int Nw = hidden * cols;
  for (int i = 0; i < hidden; ++i) {
    for (int j = 0; j < cols; ++j) {
      T w = weights[i][j];
      sum_w  += w;
      sum_abs += (w >= 0 ? w : -w);
    }
  }
  T alpha = sum_w / (T)Nw;
  T beta  = sum_abs / (T)Nw;
  if (beta < eps) beta = eps;

  T wq[hidden][cols];
#pragma HLS ARRAY_PARTITION variable=wq complete dim=2
  for (int i = 0; i < hidden; ++i) {
    for (int j = 0; j < cols; ++j) {
      T w = weights[i][j] - alpha;
      // torch.sign(0) -> 0; here we treat >=0 as +1 (difference only on exact ties)
      T s = (w >= (T)0) ? (T)1 : (T)-1;
      wq[i][j] = beta * s;
    }
  }

  const int Qb = (1 << (B - 1)) - 1;
  T xq[rows][hidden];
#pragma HLS ARRAY_PARTITION variable=xq complete dim=2
  for (int r = 0; r < rows; ++r) {
    // max |x|
    T maxabs = (T)0;
    for (int k = 0; k < hidden; ++k) {
      T v = input[r][k];
      T a = (v >= 0 ? v : -v);
      if (a > maxabs) maxabs = a;
    }
    if (maxabs < eps) maxabs = eps;
    T gamma = (T)Qb / maxabs;

    for (int k = 0; k < hidden; ++k) {
      T v = input[r][k] * gamma;
      int q = round_sym(v);
      if (q >  Qb)        q =  Qb;
      if (q < -(Qb + 1))  q = -(Qb + 1);
      xq[r][k] = (T)q / gamma; // dequantise (matches your PyTorch forward)
    }
  }

  T tmp[rows][cols];
  matmul<T, rows, hidden, cols>(xq, wq, tmp);
  for (int i = 0; i < rows; ++i) {
    vecadd<T, cols>(tmp[i], biases, result[i]);
  }
}

template<typename T, int rows, int hidden, int cols, int B = 8>
void bitlinear_ternary(
    T input[rows][hidden],
    T weights[hidden][cols],
    T biases[cols],
    T result[rows][cols],
    T eps = (T)1e-8
){
  // α unused; β = mean(|w|)
  T sum_abs = (T)0;
  const int Nw = hidden * cols;
  for (int i = 0; i < hidden; ++i)
    for (int j = 0; j < cols; ++j) {
      T w = weights[i][j];
      sum_abs += (w >= 0 ? w : -w);
    }
  T beta = sum_abs / (T)Nw;
  if (beta < eps) beta = eps;

  T wq[hidden][cols];
#pragma HLS ARRAY_PARTITION variable=wq complete dim=2
  for (int i = 0; i < hidden; ++i) {
    for (int j = 0; j < cols; ++j) {
      T t = weights[i][j] / beta;
      int r = round_sym(t);
      if (r >  1) r =  1;
      if (r < -1) r = -1;
      wq[i][j] = beta * (T)r; // ∈ {-β, 0, +β}
    }
  }

  // activations: identical to binary version
  const int Qb = (1 << (B - 1)) - 1;
  T xq[rows][hidden];
#pragma HLS ARRAY_PARTITION variable=xq complete dim=2
  for (int r = 0; r < rows; ++r) {
    T maxabs = (T)0;
    for (int k = 0; k < hidden; ++k) {
      T v = input[r][k];
      T a = (v >= 0 ? v : -v);
      if (a > maxabs) maxabs = a;
    }
    if (maxabs < eps) maxabs = eps;
    T gamma = (T)Qb / maxabs;

    for (int k = 0; k < hidden; ++k) {
      T v = input[r][k] * gamma;
      int q = round_sym(v);
      if (q >  Qb)        q =  Qb;
      if (q < -(Qb + 1))  q = -(Qb + 1);
      xq[r][k] = (T)q / gamma;
    }
  }

  T tmp[rows][cols];
  matmul<T, rows, hidden, cols>(xq, wq, tmp);
  for (int i = 0; i < rows; ++i) {
    vecadd<T, cols>(tmp[i], biases, result[i]);
  }
}

