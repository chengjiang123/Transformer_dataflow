#pragma once
template<typename T, int rows, int hidden, int cols>
void matmul(T A[rows][hidden], T B[hidden][cols], T result[rows][cols]) {
matmul_row_loop:
	for (int i = 0; i < rows; i++) {
	matmul_col_loop:
		for (int j = 0; j < cols; j++) {
			result[i][j] = 0;
		matmul_result_loop:
			for (int k = 0; k < hidden; k++) {
				result[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

template<typename T, int rows, int hidden, int cols>
void transpose_matmul(T A[rows][hidden], T B[cols][hidden], T result[rows][cols]) {
matmul_transpose_row_loop:
	for (int i = 0; i < rows; i++) {
	matmul_transpose_col_loop:
		for (int j = 0; j < cols; j++) {
			result[i][j] = 0;
		matmul_transpose_result_loop:
			for (int k = 0; k < hidden; k++) {
				result[i][j] += (T) (A[i][k] * B[j][k]);
			}
		}
	}
}

template<typename T, int rows, int hidden, int cols>
void matmul_transpose_scale(T A[rows][hidden], T B[cols][hidden], T scale_factor, T result[rows][cols]) {
matmul_transpose_scale_row_loop:
	for (int i = 0; i < rows; i++) {
	matmul_transpose_scale_col_loop:
		for (int j = 0; j < cols; j++) {
			T sum = 0;
		matmul_transpose_scale_result_loop:
			for (int k = 0; k < hidden; k++) {
				sum += (T) (A[i][k] * B[j][k]);
			}
			result[i][j] = sum / scale_factor;
		}
	}
}

template<typename T, int rows, int hidden, int cols>
void ternary_matmul(
    T A[rows][hidden],
    int8_t B[hidden][cols],  // Ternary weights: -1, 0, or 1
    T result[rows][cols]
) {
ternary_matmul_row:
    for (int i = 0; i < rows; i++) {
    ternary_matmul_col:
        for (int j = 0; j < cols; j++) {
            T sum = 0;
        ternary_matmul_k:
            for (int k = 0; k < hidden; k++) {
                #pragma HLS PIPELINE II=1
                if (B[k][j] == 1) {
                    sum += A[i][k];
                } else if (B[k][j] == -1) {
                    sum -= A[i][k];
                }
                // For 0: no operation
            }
            result[i][j] = sum;
        }
    }
}
