// Copyright 2021 Embedded Dev Research
#include <arm_neon.h>
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <omp.h>
#include "arm_gemm.hpp"

#define BLOCK_SIZE 4

void generate_matrix(std::vector<float32_t>  &A, std::vector<float32_t>  &B, uint32_t n, uint32_t m, uint32_t k) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::uniform_int_distribution<float32_t> dist {-100, 100};

    auto gen = [&dist, &mersenne_engine](){
        return dist(mersenne_engine);
    };

    A = std::vector<float32_t>(n * m);
    generate(begin(A), end(A), gen);

    B = std::vector<float32_t>(m * k);
    generate(begin(B), end(B), gen);
}

void matrix_multiply_c(const float32_t *A, const float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
    for (int i_idx = 0; i_idx < n; i_idx++) {
        for (int j_idx = 0; j_idx < m; j_idx++) {
            C[n * j_idx + i_idx] = 0;
            for (int k_idx = 0; k_idx < k; k_idx++) {
                C[n * j_idx + i_idx] += A[n * k_idx + i_idx] * B[k * j_idx + k_idx];
            }
        }
    }
}

void matrix_multiply_neon(float32_t  *A, float32_t  *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
    int A_idx;
    int B_idx;
    int C_idx;

    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;

    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;

    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;

    for (int i_idx = 0; i_idx < n; i_idx += 4) {
        for (int j_idx = 0; j_idx < m; j_idx += 4) {
            C0 = vmovq_n_f32(0);
            C1 = vmovq_n_f32(0);
            C2 = vmovq_n_f32(0);
            C3 = vmovq_n_f32(0);
            for (int k_idx = 0; k_idx < k; k_idx += 4) {
                A_idx = i_idx + n*k_idx;
                B_idx = k*j_idx + k_idx;

                A0 = vld1q_f32(A+A_idx);
                A1 = vld1q_f32(A+A_idx+n);
                A2 = vld1q_f32(A+A_idx+2*n);
                A3 = vld1q_f32(A+A_idx+3*n);

                B0 = vld1q_f32(B+B_idx);
                C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
                C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
                C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
                C0 = vfmaq_laneq_f32(C0, A3, B0, 3);

                B1 = vld1q_f32(B+B_idx+k);
                C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
                C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
                C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
                C1 = vfmaq_laneq_f32(C1, A3, B1, 3);

                B2 = vld1q_f32(B+B_idx+2*k);
                C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
                C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
                C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
                C2 = vfmaq_laneq_f32(C2, A3, B2, 3);

                B3 = vld1q_f32(B+B_idx+3*k);
                C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
                C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
                C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
                C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
            }
            C_idx = n*j_idx + i_idx;
            vst1q_f32(C+C_idx, C0);
            vst1q_f32(C+C_idx+n, C1);
            vst1q_f32(C+C_idx+2*n, C2);
            vst1q_f32(C+C_idx+3*n, C3);
        }
    }
}

bool f32comp_noteq(float32_t a, float32_t b) {
    if (fabs(a-b) < 0.000001) {
        return false;
    }
    return true;
}

bool matrix_comp(float32_t *A, float32_t *B, uint32_t rows, uint32_t cols) {
    float32_t a;
    float32_t b;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a = A[rows*j + i];
            b = B[rows*j + i];

            if (f32comp_noteq(a, b)) {
                printf("i=%d, j=%d, A=%f, B=%f\n", i, j, a, b);
                return false;
            }
        }
    }
    return true;
}
