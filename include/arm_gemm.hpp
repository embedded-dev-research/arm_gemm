// Copyright 2021 Embedded Dev Research
#ifndef INCLUDE_ARM_GEMM_HPP_
#define INCLUDE_ARM_GEMM_HPP_

#include <arm_neon.h>
#include <vector>

std::vector<double> arm_gemm(const std::vector<double>& a, const std::vector<double>& b);
void matrix_multiply_c(const float32_t *A, const float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k);
void matrix_multiply_neon(float32_t  *A, float32_t  *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k);
void matrix_multiply_4x4_neon(float32_t *A, float32_t *B, float32_t *C);
bool matrix_comp(float32_t *A, float32_t *B, uint32_t rows, uint32_t cols);
bool f32comp_noteq(float32_t a, float32_t b);

#endif  // INCLUDE_ARM_GEMM_HPP_
