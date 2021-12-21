// Copyright 2021 Embedded Dev Research
#include <iostream>
#include <arm_neon.h>
#include <chrono>
#include "arm_gemm.hpp"
#include <omp.h>

typedef std::chrono::nanoseconds ns;
typedef std::chrono::high_resolution_clock Time;

int main(int argc, char** argv) {
    for(auto count = 128; count < 1025; count *= 2) {
        std::cout << " ===== matrix size = " << count << " ===== " << std::endl;
        uint32_t n = count, m = count, k = count;
        std::vector<float32_t> a, b, c(m * k);
        generate_matrix(a, b, n, m, k);

        auto _startTime = Time::now();
        matrix_multiply_c(a.data(), b.data(), c.data(), n, m, k);
        auto _endTime = Time::now();
        auto time_c = std::chrono::duration_cast<ns>(_endTime - _startTime).count() * 0.000001;
        std::cout << "time_c = " << time_c << " ms." << std::endl;

        _startTime = Time::now();
        matrix_multiply_neon(a.data(), b.data(), c.data(), n, m, k);
        _endTime = Time::now();
        auto time_neon = std::chrono::duration_cast<ns>(_endTime - _startTime).count() * 0.000001;
        std::cout << "time_neon = " << time_neon << " ms." << std::endl;

//        _startTime = Time::now();
//        matrix_multiply_neon_omp(a.data(), b.data(), c.data(), n, m, k);
//        _endTime = Time::now();
//        auto time_neon_omp = std::chrono::duration_cast<ns>(_endTime - _startTime).count() * 0.000001;
//        std::cout << "time_neon_omp = " << time_neon_omp << " ms." << std::endl;
    }


    return 0;
}
