// Copyright 2021 Embedded Dev Research
#include "arm_gemm.hpp"

int main(int argc, char** argv) {
    std::vector<double> a;
    std::vector<double> b;
    auto c = arm_gemm(a, b);
}
