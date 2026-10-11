// Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Companion of the MMGP Optimized VRAM Allocator: raises PyTorch's own out of memory error (c10::OutOfMemoryError) for it.
// Loaded only for the PyTorch versions listed in mmgp/allocator/__init__.py (OOM_ERROR_TORCH_VERSIONS).
#include <c10/util/Exception.h>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
// The C++ runtime headers of recent toolchains read __libc_single_threaded (glibc 2.32+): defined here, always multithreaded (atomic
// reference counts), so that the library only needs the glibc of PyTorch's own wheels (manylinux 2.28)
extern "C" {
__attribute__((visibility("hidden"))) char __libc_single_threaded = 0;
}
#endif

EXPORT void mmgp_throw_out_of_memory(const char* message) {  // called by the allocator in place of its own throw: never returns
    throw c10::OutOfMemoryError({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, message);
}
