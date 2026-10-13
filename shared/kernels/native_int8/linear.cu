// Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
// Weight-only INT8: preserve activation precision and accumulate in FP32.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>
#include <type_traits>

template<class T> __device__ float as_float(T x) { return float(x); }
template<class T> __device__ T as_value(float x) { return T(x); }

template<class T> __device__ uint32_t weight_pair(uint16_t packed) {
#if __CUDA_ARCH__ >= 800
    // Encode 128+low7 (BF16) or 1024+low7 (FP16), then subtract the
    // sign-dependent offset. Every signed INT8 value is represented exactly.
    constexpr uint32_t base = std::is_same<T, __nv_bfloat16>::value ? 0x43004300u : 0x64006400u;
    uint32_t values = base | (packed & 0x7f) | ((packed & 0x7f00) << 8);
    uint32_t offsets = base | (packed & 0x80) | ((packed & 0x8000) << 8);
    if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        auto result = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&values), *reinterpret_cast<__nv_bfloat162*>(&offsets));
        return *reinterpret_cast<uint32_t*>(&result);
    } else {
        auto result = __hsub2(*reinterpret_cast<__half2*>(&values), *reinterpret_cast<__half2*>(&offsets));
        return *reinterpret_cast<uint32_t*>(&result);
    }
#else
    T lo = as_value<T>(float(int8_t(packed))), hi = as_value<T>(float(int8_t(packed >> 8)));
    return uint32_t(*reinterpret_cast<uint16_t*>(&lo)) | (uint32_t(*reinterpret_cast<uint16_t*>(&hi)) << 16);
#endif
}

template<class T> __device__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    if constexpr (std::is_same<T, __nv_bfloat16>::value)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

template<class T>
__global__ void direct_tensor_linear(const T* x, const int8_t* w, const float* s, const T* bias, T* y, int m, int n, int k, int64_t sx) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, group = lane / 4, part = lane % 4;
    int row0 = blockIdx.y * 32, col0 = blockIdx.x * 64 + warp * 16;
    float acc[2][2][4] = {};
    for (int base = 0; base < k; base += 16) {
        uint32_t a[2][4], b[2][2];
        #pragma unroll
        for (int mm = 0; mm < 2; ++mm) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int row = row0 + mm * 16 + group + (j % 2) * 8;
                int kk = base + part * 2 + (j / 2) * 8;
                a[mm][j] = row < m ? *reinterpret_cast<const uint32_t*>(x + int64_t(row) * sx + kk) : 0;
            }
        }
        #pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
            int col = col0 + nn * 8 + group;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int kk = base + part * 2 + j * 8;
                b[nn][j] = col < n ? weight_pair<T>(*reinterpret_cast<const uint16_t*>(w + int64_t(col) * k + kk)) : 0;
            }
        }
        #pragma unroll
        for (int mm = 0; mm < 2; ++mm) {
            #pragma unroll
            for (int nn = 0; nn < 2; ++nn) mma<T>(acc[mm][nn], a[mm], b[nn][0], b[nn][1]);
        }
    }
    #pragma unroll
    for (int mm = 0; mm < 2; ++mm) {
        #pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int row = row0 + mm * 16 + group + (j / 2) * 8, col = col0 + nn * 8 + part * 2 + j % 2;
                if (row < m && col < n) y[int64_t(row) * n + col] = as_value<T>(acc[mm][nn][j] * s[col] + (bias ? as_float(bias[col]) : 0.f));
            }
        }
    }
}

template<class T, int ROWS, bool MASK = false>
__global__ void gemv(const T* x, const int8_t* w, const float* scale, const T* bias, T* y,
                     int m, int n, int k, int64_t sx) {
    int lane = threadIdx.x & 31, col = blockIdx.x * 4 + threadIdx.x / 32;
    if (col >= n) return;
    float sums[ROWS][4] = {};
    for (int base = lane * 4; base < k; base += 128) {
        int packed = *reinterpret_cast<const int*>(w + int64_t(col) * k + base);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float weight = float(int8_t(packed >> (8 * j)));
            #pragma unroll
            for (int r = 0; r < ROWS; ++r)
                if (!MASK || r < m) sums[r][j] = fmaf(as_float(x[int64_t(r) * sx + base + j]), weight, sums[r][j]);
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float sum = (sums[r][0] + sums[r][1]) + (sums[r][2] + sums[r][3]);
        #pragma unroll
        for (int d = 16; d > 0; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
        if (!lane && (!MASK || r < m)) y[int64_t(r) * n + col] = as_value<T>(sum * scale[col] + (bias ? as_float(bias[col]) : 0.f));
    }
}

// A portable bounded-memory path also covers FP32 inputs and partial K tiles.
template<class T>
__global__ void scalar_linear(const T* x, const int8_t* w, const float* scale, const T* bias, T* y,
                              int m, int n, int k, int64_t sx) {
    int lane = threadIdx.x & 31, col = blockIdx.x * 4 + threadIdx.x / 32, row = blockIdx.y;
    if (col >= n) return;
    float sum = 0.f;
    for (int i = lane; i < k; i += 32) sum = fmaf(as_float(x[int64_t(row) * sx + i]), float(w[int64_t(col) * k + i]), sum);
    for (int d = 16; d > 0; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
    if (!lane) y[int64_t(row) * n + col] = as_value<T>(sum * scale[col] + (bias ? as_float(bias[col]) : 0.f));
}

template<class T>
__global__ void tensor_linear(const T* x, const int8_t* w, const float* scale, const T* bias, T* y,
                              int m, int n, int k, int64_t sx) {
#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    constexpr int BM = 32, BN = 64, BK = 128, STRIDE = BK + 8;
    __shared__ __align__(32) T a[BM * STRIDE];
    __shared__ __align__(32) T b[BN * STRIDE];
    __shared__ __align__(32) float c[BM * BN];
    int tid = threadIdx.x, warp = tid / 32, rm = (warp / 4) * 16, cn = (warp % 4) * 16;
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.f);
    for (int base = 0; base < k; base += BK) {
        for (int i = tid; i < BM * BK; i += 256) {
            int r = row0 + i / BK, kk = base + i % BK;
            a[(i / BK) * STRIDE + i % BK] = r < m && kk < k ? x[int64_t(r) * sx + kk] : as_value<T>(0.f);
        }
        for (int i = tid; i < BN * BK; i += 256) {
            int col = col0 + i / BK, kk = base + i % BK;
            b[(i / BK) * STRIDE + i % BK] = as_value<T>(col < n && kk < k ? float(w[int64_t(col) * k + kk]) : 0.f);
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::load_matrix_sync(fa, a + rm * STRIDE + kk, STRIDE);
            wmma::load_matrix_sync(fb, b + cn * STRIDE + kk, STRIDE);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(c + rm * BN + cn, fc, BN, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < BM * BN; i += 256) {
        int r = row0 + i / BN, col = col0 + i % BN;
        if (r < m && col < n) y[int64_t(r) * n + col] = as_value<T>(c[i] * scale[col] + (bias ? as_float(bias[col]) : 0.f));
    }
#endif
}

// The regular H4 Kronecker rotation used by ConvRot (not Sylvester Hadamard).
template<class T, int GROUP>
__global__ void rotate(const T* x, T* out, int k, int64_t sx) {
    __shared__ float v[GROUP];
    int i = threadIdx.x, base = blockIdx.x * GROUP, row = blockIdx.y;
    v[i] = as_float(x[int64_t(row) * sx + base + i]);
    __syncthreads();
    #pragma unroll
    for (int stride = 1; stride < GROUP; stride *= 4) {
        int group = i / (4 * stride) * (4 * stride) + i % stride;
        float s = ((v[group] + v[group + stride]) + v[group + 2 * stride]) + v[group + 3 * stride];
        float value = s - 2.f * v[group + (3 - (i / stride) % 4) * stride];
        __syncthreads();
        v[i] = value;
        __syncthreads();
    }
    out[int64_t(row) * k + base + i] = as_value<T>(v[i] * rsqrtf(float(GROUP)));
}

template<class T>
void launch(const void* xp, const int8_t* w, const float* s, const void* bp, void* yp,
            int m, int n, int k, int64_t sx, int sm, cudaStream_t stream) {
    auto x = static_cast<const T*>(xp); auto b = static_cast<const T*>(bp); auto y = static_cast<T*>(yp);
    if (k % 4 == 0 && m == 1) gemv<T, 1><<<(n + 3) / 4, 128, 0, stream>>>(x, w, s, b, y, m, n, k, sx);
    else if (k % 4 == 0 && m == 2) gemv<T, 2><<<(n + 3) / 4, 128, 0, stream>>>(x, w, s, b, y, m, n, k, sx);
    else if (k % 4 == 0 && m == 4) gemv<T, 4><<<(n + 3) / 4, 128, 0, stream>>>(x, w, s, b, y, m, n, k, sx);
    else if (k % 4 == 0 && m <= 8) gemv<T, 8, true><<<(n + 3) / 4, 128, 0, stream>>>(x, w, s, b, y, m, n, k, sx);
    else if (k % 4 == 0 && m <= 16) gemv<T, 16, true><<<(n + 3) / 4, 128, 0, stream>>>(x, w, s, b, y, m, n, k, sx);
    else if (sm >= 80 && m > 4 && k % 16 == 0 && sx % 2 == 0 && uintptr_t(x) % 4 == 0) direct_tensor_linear<T><<<dim3((n + 63) / 64, (m + 31) / 32), 128, 0, stream>>>(x, w, s, b, y, m, n, k, sx);
    else if (sm >= 80 && m > 4) tensor_linear<T><<<dim3((n + 63) / 64, (m + 31) / 32), 256, 0, stream>>>(x, w, s, b, y, m, n, k, sx);
    else scalar_linear<T><<<dim3((n + 3) / 4, m), 128, 0, stream>>>(x, w, s, b, y, m, n, k, sx);
}

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

API int wangp_int8_abi() { return 3; }

API int wangp_int8_linear(const void* x, const int8_t* w, const float* s, const void* bias, void* y,
                          int m, int n, int k, int64_t sx, int dtype, int sm, cudaStream_t stream) {
    if (!m || !n) return 0;
    if (dtype == 0) launch<__half>(x, w, s, bias, y, m, n, k, sx, sm, stream);
    else if (dtype == 1) launch<__nv_bfloat16>(x, w, s, bias, y, m, n, k, sx, sm, stream);
    else scalar_linear<float><<<dim3((n + 3) / 4, m), 128, 0, stream>>>(static_cast<const float*>(x), w, s, static_cast<const float*>(bias), static_cast<float*>(y), m, n, k, sx);
    return int(cudaGetLastError());
}

template<class T> void launch_rotation(const void* x, void* y, int m, int k, int64_t sx, int group, cudaStream_t stream) {
    if (group == 256) rotate<T, 256><<<dim3(k / 256, m), 256, 0, stream>>>(static_cast<const T*>(x), static_cast<T*>(y), k, sx);
    else if (group == 64) rotate<T, 64><<<dim3(k / 64, m), 64, 0, stream>>>(static_cast<const T*>(x), static_cast<T*>(y), k, sx);
    else rotate<T, 16><<<dim3(k / 16, m), 16, 0, stream>>>(static_cast<const T*>(x), static_cast<T*>(y), k, sx);
}

API int wangp_int8_rotate(const void* x, void* y, int m, int k, int64_t sx, int dtype, int group, cudaStream_t stream) {
    if (!m) return 0;
    if (dtype == 0) launch_rotation<__half>(x, y, m, k, sx, group, stream);
    else if (dtype == 1) launch_rotation<__nv_bfloat16>(x, y, m, k, sx, group, stream);
    else launch_rotation<float>(x, y, m, k, sx, group, stream);
    return int(cudaGetLastError());
}
