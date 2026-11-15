#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace power {

struct Options {
    int n = 256;
    int iterations = 48;
    int threads = 256;
};

inline int positive_int(const char* argument, const char* flag) {
    char* end = nullptr;
    long number = std::strtol(argument, &end, 10);
    if (argument == end || *end != '\0' || number < 1 || number > 1000000L) {
        throw std::invalid_argument(std::string("Invalid value for ") + flag + ": " + argument);
    }
    return static_cast<int>(number);
}

inline Options parse_options(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--n N] [--iterations K] [--threads T]\n"
                      << "  N: dense matrix dimension (1..4096, default 256)\n"
                      << "  K: fixed power iterations (default 48)\n"
                      << "  T: CUDA threads/block (32..1024, power of 2, default 256)\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + flag);
        if (flag == "--n") opts.n = positive_int(argv[++i], "--n");
        else if (flag == "--iterations") opts.iterations = positive_int(argv[++i], "--iterations");
        else if (flag == "--threads") opts.threads = positive_int(argv[++i], "--threads");
        else throw std::invalid_argument("Unknown flag: " + flag);
    }
    if (opts.n > 4096) throw std::invalid_argument("N > 4096 disabled to avoid excessive memory use");
    if (opts.threads < 32 || opts.threads > 1024 || (opts.threads & (opts.threads - 1)) != 0)
        throw std::invalid_argument("--threads must be a power of two in [32,1024]");
    return opts;
}

// Dense symmetric positive definite example: A = 2I + 11^T/N.
// Exact eigenvalues: lambda_max=3 (unit-constant eigenvector) and 2 otherwise.
inline std::vector<float> make_matrix(int n) {
    const size_t width = static_cast<size_t>(n);
    std::vector<float> a(width * width);
    const float shared = 1.0f / static_cast<float>(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            a[static_cast<size_t>(i) * width + static_cast<size_t>(j)] = shared + (i == j ? 2.0f : 0.0f);
    }
    return a;
}

inline std::vector<float> initial_vector(int n) {
    std::vector<float> v(static_cast<size_t>(n));
    constexpr double tau = 6.2831853071795864769;
    double norm2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double angle = tau * static_cast<double>(i) / static_cast<double>(n);
        v[i] = static_cast<float>(0.4 + 0.35 * std::sin(angle) + 0.19 * std::cos(2.0 * angle));
        norm2 += static_cast<double>(v[i]) * v[i];
    }
    const float inverse = static_cast<float>(1.0 / std::sqrt(norm2));
    for (float& x : v) x *= inverse;
    return v;
}

inline void matvec_cpu(const float* a, const float* v, float* w, int n) {
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        const size_t row = static_cast<size_t>(i) * static_cast<size_t>(n);
        for (int j = 0; j < n; ++j) sum += a[row + j] * v[j];
        w[i] = sum;
    }
}

struct Diagnostics {
    double eigenvalue;
    double relative_residual;
};

// Requires w=Av; computes the Rayleigh quotient and ||Av-lambda v||/||Av||.
inline Diagnostics diagnose(const std::vector<float>& v, const std::vector<float>& w) {
    double dot_vw = 0.0, dot_vv = 0.0, dot_ww = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        dot_vw += static_cast<double>(v[i]) * w[i];
        dot_vv += static_cast<double>(v[i]) * v[i];
        dot_ww += static_cast<double>(w[i]) * w[i];
    }
    if (dot_vv <= 0.0 || dot_ww <= 0.0) throw std::runtime_error("Degenerate eigenvector");
    double lambda = dot_vw / dot_vv;
    double residual_sq = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        double diff = static_cast<double>(w[i]) - lambda * v[i];
        residual_sq += diff * diff;
    }
    return {lambda, std::sqrt(residual_sq / dot_ww)};
}

inline void print_result(const char* backend, const Options& opts, Diagnostics d,
                         double compute_ms, double total_ms, const std::string& device = "") {
    std::cout << std::setprecision(12)
              << "RESULT backend=" << backend
              << " n=" << opts.n << " iterations=" << opts.iterations
              << " threads=" << opts.threads
              << " eigenvalue=" << d.eigenvalue
              << " eigen_error=" << std::abs(d.eigenvalue - 3.0)
              << " relative_residual=" << d.relative_residual
              << " compute_ms=" << compute_ms << " total_ms=" << total_ms;
    if (!device.empty()) std::cout << " device=" << device;
    std::cout << '\n';
}

} // namespace power
