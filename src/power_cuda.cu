#include "power_common.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <memory>
#include <string>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t error = (call);                                            \
    if (error != cudaSuccess)                                              \
        throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(error)); \
} while (0)

// One warp per row. Each lane reads strided columns and warp shuffle sums
// the contributions. Handles arbitrary N, including nonmultiples of 32.
__global__ void dense_matvec(const float* a, const float* v, float* w, int n) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int lane = threadIdx.x % warpSize;
    if (warp_id >= n) return;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(warp_id) * static_cast<size_t>(n);
    for (int j = lane; j < n; j += warpSize)
        sum += a[base + static_cast<size_t>(j)] * v[j];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) w[warp_id] = sum;
}

// Each CTA reduces one partial sum then atomically accumulates to scalar.
// The scalar must be zeroed before every launch.
__global__ void squared_norm(const float* w, float* total, int n) {
    extern __shared__ float scratch[];
    const int tid = threadIdx.x;
    const int index = blockIdx.x * blockDim.x + tid;
    float x = index < n ? w[index] : 0.0f;
    scratch[tid] = x * x;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(total, scratch[0]);
}

__global__ void normalize_vector(const float* w, const float* squared, float* v, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) v[index] = w[index] * rsqrtf(*squared);
}

struct DeviceFloat {
    float* ptr = nullptr;
    explicit DeviceFloat(size_t count) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float))); }
    ~DeviceFloat() { if (ptr) cudaFree(ptr); }
    DeviceFloat(const DeviceFloat&) = delete;
    DeviceFloat& operator=(const DeviceFloat&) = delete;
};

struct Event {
    cudaEvent_t handle{};
    Event() { CUDA_CHECK(cudaEventCreate(&handle)); }
    ~Event() { cudaEventDestroy(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

int main(int argc, char** argv) {
    try {
        const auto opts = power::parse_options(argc, argv);
        int device_count = 0;
        const cudaError_t probe = cudaGetDeviceCount(&device_count);
        if (probe != cudaSuccess || device_count < 1) {
            std::cerr << "GPU_UNAVAILABLE: no CUDA-capable GPU detected: "
                      << cudaGetErrorString(probe) << '\n';
            return 3;
        }
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        if (opts.threads > properties.maxThreadsPerBlock)
            throw std::invalid_argument("Block size exceeds device's maximum threads/block");

        auto matrix = power::make_matrix(opts.n);
        auto v = power::initial_vector(opts.n);
        std::vector<float> w(static_cast<size_t>(opts.n));
        const size_t nn = static_cast<size_t>(opts.n);

        const auto total_start = std::chrono::steady_clock::now();
        DeviceFloat d_a(nn * nn), d_v(nn), d_w(nn), d_norm(1);
        CUDA_CHECK(cudaMemcpy(d_a.ptr, matrix.data(), nn * nn * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v.ptr, v.data(), nn * sizeof(float), cudaMemcpyHostToDevice));

        const int warps_per_block = opts.threads / 32;
        const int mat_blocks = (opts.n + warps_per_block - 1) / warps_per_block;
        const int vector_blocks = (opts.n + opts.threads - 1) / opts.threads;
        Event start, stop;
        CUDA_CHECK(cudaEventRecord(start.handle));
        for (int step = 0; step < opts.iterations; ++step) {
            dense_matvec<<<mat_blocks, opts.threads>>>(d_a.ptr, d_v.ptr, d_w.ptr, opts.n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemsetAsync(d_norm.ptr, 0, sizeof(float)));
            squared_norm<<<vector_blocks, opts.threads, opts.threads * sizeof(float)>>>(d_w.ptr, d_norm.ptr, opts.n);
            CUDA_CHECK(cudaGetLastError());
            normalize_vector<<<vector_blocks, opts.threads>>>(d_w.ptr, d_norm.ptr, d_v.ptr, opts.n);
            CUDA_CHECK(cudaGetLastError());
        }
        dense_matvec<<<mat_blocks, opts.threads>>>(d_a.ptr, d_v.ptr, d_w.ptr, opts.n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop.handle));
        CUDA_CHECK(cudaEventSynchronize(stop.handle));
        float compute_ms_float = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&compute_ms_float, start.handle, stop.handle));

        CUDA_CHECK(cudaMemcpy(v.data(), d_v.ptr, nn * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(w.data(), d_w.ptr, nn * sizeof(float), cudaMemcpyDeviceToHost));
        const auto total_end = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        auto diagnostics = power::diagnose(v, w);
        std::string name(properties.name);
        std::replace(name.begin(), name.end(), ' ', '_');
        power::print_result("cuda", opts, diagnostics, static_cast<double>(compute_ms_float), total_ms, name);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << '\n';
        return 1;
    }
}
