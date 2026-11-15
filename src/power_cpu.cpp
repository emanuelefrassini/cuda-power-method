#include "power_common.hpp"
#include <chrono>

int main(int argc, char** argv) {
    try {
        const auto opts = power::parse_options(argc, argv);
        const auto matrix = power::make_matrix(opts.n);
        auto v = power::initial_vector(opts.n);
        std::vector<float> w(static_cast<size_t>(opts.n), 0.0f);

        const auto start = std::chrono::steady_clock::now();
        for (int k = 0; k < opts.iterations; ++k) {
            power::matvec_cpu(matrix.data(), v.data(), w.data(), opts.n);
            double norm2 = 0.0;
            for (float z : w) norm2 += static_cast<double>(z) * z;
            if (!(norm2 > 0.0)) throw std::runtime_error("Zero power-iteration norm");
            const float invnorm = static_cast<float>(1.0 / std::sqrt(norm2));
            for (int j = 0; j < opts.n; ++j) v[j] = w[j] * invnorm;
        }
        power::matvec_cpu(matrix.data(), v.data(), w.data(), opts.n);
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        const auto diagnostics = power::diagnose(v, w);
        power::print_result("cpu", opts, diagnostics, elapsed_ms, elapsed_ms);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << '\n';
        return 1;
    }
}
