// Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// MMGP Optimized VRAM Allocator: a CUDA pluggable allocator for PyTorch (Windows, Linux), built on the CUDA driver API loaded at run time.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <deque>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <thread>
#define EXPORT extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#include <pthread.h>
#define EXPORT extern "C" __attribute__((visibility("default")))
#if defined(__x86_64__) && defined(__GLIBC__)
// The oldest versions of the glibc symbols used, so that the library loads with glibc 2.17 (manylinux2014) whatever glibc built it
__asm__(".symver dlopen,dlopen@GLIBC_2.2.5");
__asm__(".symver dlsym,dlsym@GLIBC_2.2.5");
__asm__(".symver pthread_create,pthread_create@GLIBC_2.2.5");
__asm__(".symver pthread_detach,pthread_detach@GLIBC_2.2.5");
// The static C++ runtime of recent toolchains reads __libc_single_threaded (glibc 2.32+): always multithreaded here (atomic reference
// counts). Its thread keys and once-calls are linked with --wrap (build.py) to the glibc 2.2.5 versions of the same functions.
extern "C" {
__attribute__((visibility("hidden"))) char __libc_single_threaded = 0;
int compat_pthread_key_create(pthread_key_t*, void (*)(void*));
int compat_pthread_key_delete(pthread_key_t);
void* compat_pthread_getspecific(pthread_key_t);
int compat_pthread_setspecific(pthread_key_t, const void*);
int compat_pthread_once(pthread_once_t*, void (*)(void));
__attribute__((visibility("hidden"))) int __wrap_pthread_key_create(pthread_key_t* key, void (*destructor)(void*)) { return compat_pthread_key_create(key, destructor); }
__attribute__((visibility("hidden"))) int __wrap___pthread_key_create(pthread_key_t* key, void (*destructor)(void*)) { return compat_pthread_key_create(key, destructor); }
__attribute__((visibility("hidden"))) int __wrap_pthread_key_delete(pthread_key_t key) { return compat_pthread_key_delete(key); }
__attribute__((visibility("hidden"))) void* __wrap_pthread_getspecific(pthread_key_t key) { return compat_pthread_getspecific(key); }
__attribute__((visibility("hidden"))) int __wrap_pthread_setspecific(pthread_key_t key, const void* value) { return compat_pthread_setspecific(key, value); }
__attribute__((visibility("hidden"))) int __wrap_pthread_once(pthread_once_t* once, void (*routine)(void)) { return compat_pthread_once(once, routine); }
}
__asm__(".symver compat_pthread_key_create,pthread_key_create@GLIBC_2.2.5");
__asm__(".symver compat_pthread_key_delete,pthread_key_delete@GLIBC_2.2.5");
__asm__(".symver compat_pthread_getspecific,pthread_getspecific@GLIBC_2.2.5");
__asm__(".symver compat_pthread_setspecific,pthread_setspecific@GLIBC_2.2.5");
__asm__(".symver compat_pthread_once,pthread_once@GLIBC_2.2.5");
#endif
#endif

// The subset of the CUDA driver API used here, with the layouts and values of cuda.h (stable driver ABI).
typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;
typedef struct CUctx_st* CUcontext;
typedef struct CUmemPoolHandle_st* CUmemoryPool;
struct CUmemLocation {
    int type;
    int id;
};
struct CUmemAllocationProp {
    int type;
    int requestedHandleTypes;
    CUmemLocation location;
    void* win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
};
struct CUmemAccessDesc {
    CUmemLocation location;
    int flags;
};
struct CUmemPoolProps {
    int allocType;
    int handleTypes;
    CUmemLocation location;
    void* win32SecurityAttributes;
    unsigned char reserved[64];  // maxSize and usage on recent drivers: 0 selects their defaults
};
enum : int {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CU_MEM_ALLOCATION_TYPE_PINNED = 1,
    CU_MEM_LOCATION_TYPE_DEVICE = 1,
    CU_MEM_LOCATION_TYPE_HOST = 2,
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3,
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4,
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5,
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7,
    CU_STREAM_CAPTURE_STATUS_NONE = 0,
    CU_STREAM_CAPTURE_MODE_RELAXED = 2,
    CU_EVENT_DISABLE_TIMING = 2,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
};

namespace {
struct Driver {
    CUresult (*Init)(unsigned);
    CUresult (*DeviceGet)(CUdevice*, int);
    CUresult (*DeviceGetAttribute)(int*, int, CUdevice);
    CUresult (*DeviceGetDefaultMemPool)(CUmemoryPool*, CUdevice);
    CUresult (*CtxGetCurrent)(CUcontext*);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*CtxSynchronize)();
    CUresult (*MemPoolSetAttribute)(CUmemoryPool, int, void*);
    CUresult (*MemPoolGetAttribute)(CUmemoryPool, int, void*);
    CUresult (*MemPoolTrimTo)(CUmemoryPool, size_t);
    CUresult (*MemPoolCreate)(CUmemoryPool*, const CUmemPoolProps*);
    CUresult (*MemAllocFromPoolAsync)(CUdeviceptr*, size_t, CUmemoryPool, CUstream);
    CUresult (*MemFreeAsync)(CUdeviceptr, CUstream);
    CUresult (*MemAlloc)(CUdeviceptr*, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemCreate)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
    CUresult (*MemRelease)(CUmemGenericAllocationHandle);
    CUresult (*MemAddressReserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
    CUresult (*MemAddressFree)(CUdeviceptr, size_t);
    CUresult (*MemMap)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    CUresult (*MemUnmap)(CUdeviceptr, size_t);
    CUresult (*MemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
    CUresult (*EventCreate)(CUevent*, unsigned);
    CUresult (*EventRecord)(CUevent, CUstream);
    CUresult (*EventQuery)(CUevent);
    CUresult (*EventSynchronize)(CUevent);
    CUresult (*EventDestroy)(CUevent);
    CUresult (*StreamWaitEvent)(CUstream, CUevent, unsigned);
    CUresult (*StreamIsCapturing)(CUstream, int*);
    CUresult (*ThreadExchangeStreamCaptureMode)(int*);
    CUresult (*MemGetInfo)(size_t*, size_t*);
    CUresult (*MemGetAllocationGranularity)(size_t*, const CUmemAllocationProp*, int);
    CUresult (*DeviceGetPCIBusId)(char*, int, CUdevice);
} cu;

struct NvmlMemory {
    unsigned long long total, free, used;
};
struct Nvml {  // NVML, installed with the NVIDIA driver: free memory of the whole GPU, all processes included
    int (*Init)();
    int (*DeviceGetHandleByPciBusId)(const char*, void**);
    int (*DeviceGetMemoryInfo)(void*, NvmlMemory*);
} nvml;

int driver_status = -1;  // -1 not loaded, 0 loaded, 1 library missing, 2 entry point missing, 3 cuInit failed

int load_driver() {
    if (driver_status >= 0) return driver_status;
#ifdef _WIN32
    void* library = reinterpret_cast<void*>(LoadLibraryA("nvcuda.dll"));
    auto symbol = [library](const char* name) { return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(library), name)); };
#else
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    auto symbol = [library](const char* name) { return dlsym(library, name); };
#endif
    if (!library) return driver_status = 1;
    bool complete = true;
    auto bind = [&](auto& function, const char* name) {
        function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(symbol(name));
        complete = complete && function != nullptr;
    };
    // the unversioned names of the legacy default stream entry points, as cuda.h without per-thread default streams (PyTorch's)
    bind(cu.Init, "cuInit");
    bind(cu.DeviceGet, "cuDeviceGet");
    bind(cu.DeviceGetAttribute, "cuDeviceGetAttribute");
    bind(cu.DeviceGetDefaultMemPool, "cuDeviceGetDefaultMemPool");
    bind(cu.CtxGetCurrent, "cuCtxGetCurrent");
    bind(cu.CtxSetCurrent, "cuCtxSetCurrent");
    bind(cu.CtxSynchronize, "cuCtxSynchronize");
    bind(cu.MemPoolSetAttribute, "cuMemPoolSetAttribute");
    bind(cu.MemPoolGetAttribute, "cuMemPoolGetAttribute");
    bind(cu.MemPoolTrimTo, "cuMemPoolTrimTo");
    bind(cu.MemPoolCreate, "cuMemPoolCreate");
    bind(cu.MemAllocFromPoolAsync, "cuMemAllocFromPoolAsync");
    bind(cu.MemFreeAsync, "cuMemFreeAsync");
    bind(cu.MemAlloc, "cuMemAlloc_v2");
    bind(cu.MemFree, "cuMemFree_v2");
    bind(cu.MemCreate, "cuMemCreate");
    bind(cu.MemRelease, "cuMemRelease");
    bind(cu.MemAddressReserve, "cuMemAddressReserve");
    bind(cu.MemAddressFree, "cuMemAddressFree");
    bind(cu.MemMap, "cuMemMap");
    bind(cu.MemUnmap, "cuMemUnmap");
    bind(cu.MemSetAccess, "cuMemSetAccess");
    bind(cu.EventCreate, "cuEventCreate");
    bind(cu.EventRecord, "cuEventRecord");
    bind(cu.EventQuery, "cuEventQuery");
    bind(cu.EventSynchronize, "cuEventSynchronize");
    bind(cu.EventDestroy, "cuEventDestroy_v2");
    bind(cu.StreamWaitEvent, "cuStreamWaitEvent");
    bind(cu.StreamIsCapturing, "cuStreamIsCapturing");
    bind(cu.ThreadExchangeStreamCaptureMode, "cuThreadExchangeStreamCaptureMode");
    bind(cu.MemGetInfo, "cuMemGetInfo_v2");
    bind(cu.MemGetAllocationGranularity, "cuMemGetAllocationGranularity");
    bind(cu.DeviceGetPCIBusId, "cuDeviceGetPCIBusId");
    if (!complete) return driver_status = 2;
    return driver_status = cu.Init(0) == CUDA_SUCCESS ? 0 : 3;
}

bool load_nvml() {  // without NVML (not found), only the process' own budget limits the allocator
#ifdef _WIN32
    void* library = reinterpret_cast<void*>(LoadLibraryA("nvml.dll"));
    auto symbol = [library](const char* name) { return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(library), name)); };
#else
    void* library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    auto symbol = [library](const char* name) { return dlsym(library, name); };
#endif
    if (!library) return false;
    nvml.Init = reinterpret_cast<int (*)()>(symbol("nvmlInit_v2"));
    nvml.DeviceGetHandleByPciBusId = reinterpret_cast<int (*)(const char*, void**)>(symbol("nvmlDeviceGetHandleByPciBusId_v2"));
    nvml.DeviceGetMemoryInfo = reinterpret_cast<int (*)(void*, NvmlMemory*)>(symbol("nvmlDeviceGetMemoryInfo"));
    return nvml.Init && nvml.DeviceGetHandleByPciBusId && nvml.DeviceGetMemoryInfo && nvml.Init() == 0;
}

using Event = std::shared_ptr<CUevent_st>;  // recorded when memory was freed, shared by its chunks until they are reused
using MempoolId = std::pair<unsigned long long, unsigned long long>;  // c10::cuda::MempoolId_t
using CaptureFilter = std::function<bool(CUstream)>;                  // true for the streams that allocate to a capture's pool

struct Range {
    CUdeviceptr va = 0;
    size_t size = 0;  // chunks * chunk size
    std::vector<CUmemGenericAllocationHandle> chunks;
    CUstream stream = nullptr;  // stream of the allocation
    Event freed;
};
struct Spare {  // a chunk mapped nowhere, free once its event has completed
    CUmemGenericAllocationHandle handle;
    Event freed;
    CUstream stream;  // stream of the range it came from
};
struct Stale {  // a virtual range whose chunks moved to other ranges, unmapped once the work queued before its free has completed
    CUdeviceptr va;
    size_t size;
    Event freed;
    CUcontext context;
};
struct Small {  // a live allocation of the driver pool
    size_t size;  // as allocated (cached sizes are rounded)
    CUstream stream;
};
struct Spilled {  // spill mode: a tensor VRAM had no room for, in pinned system RAM mapped into the GPU's address space
    size_t size;  // mapped size
    CUstream stream;
    Event freed;  // once freed: kept mapped for the next spill of the same size
    bool driver = false;  // allocated by the driver as PyTorch's allocator does (alloc_driver): given back to the driver once freed
};
struct Pool {  // private memory of CUDA graphs, kept while a graph may replay
    int use_count = 0;
    size_t live = 0;
    std::unordered_map<CUdeviceptr, size_t> blocks;  // every allocation of the pool
    std::multimap<size_t, CUdeviceptr> free_blocks;  // the free ones, by size
};
struct Capture {  // between PyTorch's begin and end of allocation to a pool
    MempoolId id;
    CaptureFilter filter;
};
// Debug mode (mmgp/allocator/debug.py): the allocations of debug_threshold bytes and more are recorded with an origin that a Python
// callback returns (module, tag, stack), the records are copied at the peak of each phase and when an allocation fails, and the
// allocations of each origin are totalled. A peak is copied lazily, at the first free after a new maximum, while the memory in use is
// still at that maximum. Off (threshold 0): one test in each allocation and free.
struct DebugEntry {  // also the layout exported to Python
    uint64_t ptr;
    int64_t size, origin, seq, kind;  // kind: 0 chunk range, 1 small pool, 2 mid-size pool, 3 spilled, 4 graph pool, 5 driver spill
    double time;                      // seconds since the recording started
};
struct DebugHeader {  // exported layout
    int64_t phase, allocated, reserved, spilled, chunks, small_pool, mid_pool, graph_pools, cached_ranges, spare, small_cached, request, entries;
    double time;
};
struct DebugSnapshot {
    DebugHeader header = {};
    std::vector<DebugEntry> entries;
};
struct OriginTotals {  // exported as origin, count, bytes, max size, freed, lifetime of the freed ones in microseconds
    int64_t count = 0, bytes = 0, max_size = 0, freed = 0, life_us = 0;
};

struct Device {
    CUmemAllocationProp prop = {};
    CUmemAccessDesc access = {};
    CUmemoryPool small_pool = nullptr;  // the device's default pool, for allocations under mid_threshold
    CUmemoryPool mid_pool = nullptr;    // from mid_threshold up to large_threshold
    CUcontext context = nullptr;
    std::unordered_map<CUdeviceptr, Range> live;
    std::deque<Range> cached;  // oldest first
    std::vector<Spare> spare;
    std::unordered_map<CUdeviceptr, Small> live_small;
    std::unordered_map<CUstream, std::unordered_map<size_t, std::vector<CUdeviceptr>>> small_cache;
    std::unordered_map<CUdeviceptr, std::vector<CUstream>> stream_uses;  // other streams that used a live allocation (record_stream)
    std::map<MempoolId, Pool> pools;
    std::unordered_map<CUdeviceptr, MempoolId> pool_of;  // live pool allocations
    std::vector<Capture> captures;
    std::vector<CUdeviceptr> deferred;  // frees postponed to the end of the captures
    std::deque<Event> probes;  // the events of the latest large frees, oldest first
    int64_t allocated = 0, peak_allocated = 0, chunks_total = 0, peak_reserved = 0, small_calls = 0, small_cached = 0, pools_bytes = 0;
    int64_t mark_peak = 0;  // peak of the allocated memory since vmm_mark_reset, apart from the peaks torch.cuda.reset_peak_memory_stats resets
    int64_t recoveries = 0;  // allocations that found the VRAM short: they waited for other streams or synchronized the device to reuse memory
    unsigned long long small_seen = 0, mid_seen = 0;  // reserved sizes of the driver pools at the last check, to notice their growth
    std::unordered_map<CUdeviceptr, Spilled> live_spilled;
    std::vector<std::pair<CUdeviceptr, Spilled>> spill_cache;  // freed spilled tensors, oldest first
    size_t host_granularity = 0;                               // 0 when the driver cannot map system RAM this way
    int64_t spilled_bytes = 0;                                 // system RAM mapped by spills, live and cached
    int64_t driver_spilled = 0;                                // live allocations of the driver (alloc_driver)
    void* nvml_device = nullptr;                               // see vram_short
    int64_t nvml_free = 0, nvml_reserved = 0;                  // whole GPU free memory at the last NVML reading, reserved_now then
    std::chrono::steady_clock::time_point nvml_read{};
    bool beyond_gpu = false;                                   // vmm_alloc's last attempt: the VRAM of other processes is not counted
    int refusal = 0;                                           // what refused new memory last (Refusal), for the out of memory message
    int64_t refused_need = 0, refused_headroom = 0;            // the new memory it needed, headroom included, and that headroom
    std::unordered_map<CUdeviceptr, DebugEntry> debug_live;  // debug mode: the live allocations of the threshold and more
    std::map<int64_t, DebugSnapshot> debug_peaks;            // per phase
    DebugSnapshot debug_oom;                                 // when the last allocation failed
    bool debug_pending = false;                              // a new maximum of the phase is not copied yet
    std::unordered_map<int64_t, OriginTotals> debug_origins;
};

// Never destroyed: the detached unmapping thread still waits on them while the process exits (on Linux, destroying a condition variable
// with a waiter blocks the exit), and the driver may already be shut down when static destructors would release events and memory.
std::mutex& lock = *new std::mutex;
size_t large_threshold = 256ull << 20, chunk_size = 32ull << 20;
size_t mid_threshold = 4ull << 20;
size_t vram_headroom = 256ull << 20;  // left free by chunks and the mid-size pool: driver (kernel modules, local memory), other libraries
size_t small_headroom = 64ull << 20;  // left free by the small blocks, which may use the rest (spilled tensors still need small ones)
bool spill = false;  // allocations that VRAM has no room for go to pinned system RAM instead of failing
bool driver_spill = false;  // and when RAM is short for that, to the driver's own allocation, as with PyTorch's allocator (alloc_driver)
int64_t vram_limit = 0;  // vmm_set_vram_limit: VRAM this process may reserve, 0 for the GPU's (emulates a smaller GPU: a hard limit, unlike
                         // another process holding VRAM, which Windows pages out of the GPU when it is idle)
std::atomic<size_t> debug_threshold{0};
int64_t (*debug_origin)(size_t, int) = nullptr;  // set once, never cleared: a thread may still call it while a recording stops
int64_t (*pressure_callback)(size_t, int) = nullptr;  // vmm_set_pressure_callback: frees memory the application can do without
void (*oom_thrower)(const char*) = nullptr;  // vmm_set_oom_thrower: throws PyTorch's own out of memory error (oom_error.cpp)
thread_local bool in_pressure_callback = false;
thread_local bool in_debug_origin = false;
int64_t debug_phase = 0, debug_seq = 0;
std::chrono::steady_clock::time_point debug_t0;
size_t small_cache_max = 1ull << 20, small_cache_limit = 256ull << 20;  // sizes cached per stream, total cached per device
std::unordered_map<int, Device>& devices = *new std::unordered_map<int, Device>;
std::mutex& retired_lock = *new std::mutex;  // stale ranges are unmapped by a background thread: unmapping takes about 1 ms per GB on Windows
std::condition_variable& retired_ready = *new std::condition_variable;
std::condition_variable& retired_unmapped = *new std::condition_variable;  // notified when no stale range is left to unmap
std::deque<Stale>& retired = *new std::deque<Stale>;
size_t unmaps_pending = 0;  // stale ranges queued or being unmapped
bool unmapper_started = false;

struct RelaxedCapture {  // allocation calls stay allowed in this thread while a stream captures, as PyTorch does for cudaMalloc
    int mode = CU_STREAM_CAPTURE_MODE_RELAXED;
    bool active;
    explicit RelaxedCapture(bool needed) : active(needed) {
        if (active) cu.ThreadExchangeStreamCaptureMode(&mode);
    }
    ~RelaxedCapture() {
        if (active) cu.ThreadExchangeStreamCaptureMode(&mode);
    }
};

bool any_capture() {
    for (auto& entry : devices)
        if (!entry.second.captures.empty()) return true;
    return false;
}

Event record_event(CUstream stream) {
    CUevent event = nullptr;
    cu.EventCreate(&event, CU_EVENT_DISABLE_TIMING);
    cu.EventRecord(event, stream);
    return Event(event, [](CUevent e) { cu.EventDestroy(e); });
}

void wait(CUstream stream, const Event& event) {
    if (event) cu.StreamWaitEvent(stream, event.get(), 0);
}

bool capturing(CUstream stream) {
    int status = CU_STREAM_CAPTURE_STATUS_NONE;
    cu.StreamIsCapturing(stream, &status);
    return status != CU_STREAM_CAPTURE_STATUS_NONE;
}

unsigned long long pool_attribute(CUmemoryPool pool, int attribute) {
    unsigned long long value = 0;
    if (pool) cu.MemPoolGetAttribute(pool, attribute, &value);
    return value;
}

int64_t reserved_now(const Device& d) {
    unsigned long long driver_pools = pool_attribute(d.small_pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT) + pool_attribute(d.mid_pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT);
    return d.chunks_total * static_cast<int64_t>(chunk_size) + static_cast<int64_t>(driver_pools) + d.pools_bytes;
}

void trim_pools(Device& d) {
    cu.MemPoolTrimTo(d.small_pool, 0);
    cu.MemPoolTrimTo(d.mid_pool, 0);
    d.small_seen = pool_attribute(d.small_pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT);
    d.mid_seen = pool_attribute(d.mid_pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT);
}

// With Windows' driver model (also under WSL) the driver commits memory beyond the GPU's own, in shared GPU memory, and can fail later
// when it cannot make it resident: that error is permanent, the CUDA context is lost with all its VRAM. New memory is only taken while
// the GPU has room for it (and the headroom); otherwise the allocation fails, which raises an out of memory error, or in spill mode
// spills (alloc_spilled, then as a last resort alloc_driver, which accepts that risk as PyTorch's allocator does).
// The free VRAM the driver reports is this process' budget, which on Windows ignores the VRAM used by other processes (a second WanGP,
// Deepy, ComfyUI, a game): NVML's free memory of the whole GPU is checked too. NVML only counts memory once written: it is read at most
// every 2 s, when this process' allocations have normally been written, and the allocator's own growth since is subtracted.
// An allocation that would fail anyway is tried once more without it (beyond_gpu): Windows pages the VRAM of idle processes out.
enum Refusal : int { REFUSED_BY_BUDGET = 1, REFUSED_BY_GPU, REFUSED_BY_LIMIT, REFUSED_BY_DRIVER };

int64_t room_now(Device& d, int* bound = nullptr) {  // bound: the Refusal of the smallest of the limits
    size_t free_bytes = 0, total = 0;
    cu.MemGetInfo(&free_bytes, &total);
    int64_t room = static_cast<int64_t>(free_bytes);
    int smallest = REFUSED_BY_BUDGET;
    if (d.nvml_device && !d.beyond_gpu) {
        int64_t reserved = reserved_now(d);
        auto now = std::chrono::steady_clock::now();
        NvmlMemory info;
        if (now - d.nvml_read > std::chrono::seconds(2) && nvml.DeviceGetMemoryInfo(d.nvml_device, &info) == 0) {
            d.nvml_free = static_cast<int64_t>(info.free);
            d.nvml_reserved = reserved;
            d.nvml_read = now;
        }
        if (d.nvml_read != std::chrono::steady_clock::time_point{} && d.nvml_free - (reserved - d.nvml_reserved) < room) {
            room = d.nvml_free - (reserved - d.nvml_reserved);
            smallest = REFUSED_BY_GPU;
        }
    }
    if (vram_limit && vram_limit - reserved_now(d) < room) {
        room = vram_limit - reserved_now(d);
        smallest = REFUSED_BY_LIMIT;
    }
    if (bound) *bound = smallest;
    return room;
}

void refuse(Device& d, int refusal, size_t bytes, size_t headroom) {
    d.refusal = refusal;
    d.refused_need = static_cast<int64_t>(bytes + headroom);
    d.refused_headroom = static_cast<int64_t>(headroom);
}

bool vram_short(Device& d, size_t bytes, size_t headroom = vram_headroom) {
    int bound;
    if (room_now(d, &bound) >= static_cast<int64_t>(bytes + headroom)) return false;
    refuse(d, bound, bytes, headroom);
    return true;
}

CUresult create_chunk(Device& d, CUmemGenericAllocationHandle* handle, size_t missing) {
    // the free VRAM the driver reports only counts chunks once mapped: the room is checked for all the chunks the range still misses
    if (vram_short(d, missing * chunk_size)) return CUDA_ERROR_OUT_OF_MEMORY;
    CUresult result = cu.MemCreate(handle, chunk_size, &d.prop, 0);
    if (result != CUDA_SUCCESS) refuse(d, REFUSED_BY_DRIVER, missing * chunk_size, vram_headroom);
    return result;
}

CUresult pool_alloc(Device& d, CUdeviceptr* dptr, size_t size, CUmemoryPool pool, CUstream stream) {
    // a driver pool may only grow while the GPU has room: checked before, as a block given back after growing past the room stays reserved
    // in the pool, where the next allocation would take it; and after, for a growth that fragmentation caused
    size_t headroom = pool == d.mid_pool ? vram_headroom : small_headroom;
    unsigned long long& seen = pool == d.mid_pool ? d.mid_seen : d.small_seen;
    unsigned long long reserved = pool_attribute(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT);
    if (reserved - pool_attribute(pool, CU_MEMPOOL_ATTR_USED_MEM_CURRENT) < size && vram_short(d, size, headroom)) return CUDA_ERROR_OUT_OF_MEMORY;
    CUresult result = cu.MemAllocFromPoolAsync(dptr, size, pool, stream);
    if (result != CUDA_SUCCESS) refuse(d, REFUSED_BY_DRIVER, size, headroom);
    reserved = pool_attribute(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT);
    bool grew = reserved > seen;
    seen = reserved;
    if (result == CUDA_SUCCESS && grew && vram_short(d, 0, headroom)) {  // the pool grew past the GPU's room: the block goes back, the next trim releases it
        d.refused_need += static_cast<int64_t>(size);
        cu.MemFreeAsync(*dptr, stream);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return result;
}

void count(Device& d, int64_t delta) {
    d.allocated += delta;
    d.peak_allocated = std::max(d.peak_allocated, d.allocated);
    d.mark_peak = std::max(d.mark_peak, d.allocated);
}

void unmap_retired() {
    int mode = CU_STREAM_CAPTURE_MODE_RELAXED;  // never invalidates the CUDA graph captures of other threads
    cu.ThreadExchangeStreamCaptureMode(&mode);
    CUcontext current = nullptr;
    std::unique_lock<std::mutex> guard(retired_lock);
    for (;;) {
        retired_ready.wait(guard, [] { return !retired.empty(); });
        Stale range = std::move(retired.front());
        retired.pop_front();
        guard.unlock();
        if (range.context != current) cu.CtxSetCurrent(current = range.context);
        if (range.freed) cu.EventSynchronize(range.freed.get());
        cu.MemUnmap(range.va, range.size);
        cu.MemAddressFree(range.va, range.size);
        range.freed.reset();
        guard.lock();
        if (--unmaps_pending == 0) retired_unmapped.notify_all();
    }
}

void retire(Stale range) {
    {
        std::lock_guard<std::mutex> guard(retired_lock);
        retired.push_back(std::move(range));
        ++unmaps_pending;
    }
    retired_ready.notify_one();
}

Device& device_state(int index) {  // created by the first allocation on a device, with its context current
    auto found = devices.find(index);
    if (found != devices.end()) return found->second;
    load_driver();
    Device& d = devices[index];
    d.prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    d.prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.prop.location.id = index;
    d.access.location = d.prop.location;
    d.access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUdevice dev;
    cu.DeviceGet(&dev, index);
    cu.DeviceGetDefaultMemPool(&d.small_pool, dev);
    CUmemPoolProps mid_props = {};
    mid_props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    mid_props.location = d.prop.location;
    cu.MemPoolCreate(&d.mid_pool, &mid_props);
    CUmemAllocationProp host = d.prop;
    host.location = {CU_MEM_LOCATION_TYPE_HOST, 0};
    if (cu.MemGetAllocationGranularity(&d.host_granularity, &host, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) d.host_granularity = 0;
    char bus_id[32] = {};
    if (load_nvml() && cu.DeviceGetPCIBusId(bus_id, sizeof(bus_id), dev) == CUDA_SUCCESS && nvml.DeviceGetHandleByPciBusId(bus_id, &d.nvml_device) != 0)
        d.nvml_device = nullptr;
    for (CUmemoryPool pool : {d.small_pool, d.mid_pool}) {
        unsigned long long keep = ~0ull;  // the pools keep their memory between synchronizations, like a caching allocator
        cu.MemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &keep);
        int no = 0;  // like PyTorch, memory freed on another stream is not reused before that stream has finished with it
        cu.MemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &no);
    }
    cu.CtxGetCurrent(&d.context);
    if (!unmapper_started) {
#ifdef _WIN32
        std::thread(unmap_retired).detach();
#else
        pthread_t thread;
        pthread_create(&thread, nullptr, [](void*) -> void* { unmap_retired(); return nullptr; }, nullptr);
        pthread_detach(thread);
#endif
        unmapper_started = true;
    }
    return d;
}

size_t small_size(size_t size) {  // the smallest sizes are rounded so that cached blocks serve similar requests
    return size <= small_cache_max ? (size + 511) & ~size_t(511) : size;
}

void free_small_cache(Device& d) {  // after a synchronization: no work is pending on cached blocks
    for (auto& lists : d.small_cache)
        for (auto& sized : lists.second)
            for (auto ptr : sized.second) cu.MemFreeAsync(ptr, nullptr);
    d.small_cache.clear();
    if (d.small_cached) cu.CtxSynchronize();  // the driver pools only give back the frees a synchronization has observed
    d.small_cached = 0;
}

void release_cached_memory(Device& d) {  // after a device synchronization: everything free goes back to the driver
    {   // chunks may still be mapped by stale ranges: unmapping or releasing them while the thread unmaps those corrupts the driver's heap
        std::unique_lock<std::mutex> guard(retired_lock);
        retired_unmapped.wait(guard, [] { return unmaps_pending == 0; });
    }
    for (auto& range : d.cached) {
        cu.MemUnmap(range.va, range.size);
        cu.MemAddressFree(range.va, range.size);
        for (auto h : range.chunks) cu.MemRelease(h);
        d.chunks_total -= static_cast<int64_t>(range.chunks.size());
    }
    d.cached.clear();
    for (auto& s : d.spare) cu.MemRelease(s.handle);
    d.chunks_total -= static_cast<int64_t>(d.spare.size());
    d.spare.clear();
    free_small_cache(d);
    trim_pools(d);
}

bool reusable(const Event& freed, CUstream owner, CUstream stream) {  // without making the stream wait for work queued on another stream
    return owner == stream || !freed || cu.EventQuery(freed.get()) == CUDA_SUCCESS;
}

// Spilled tensors are pinned: system RAM that nothing else can use and the OS cannot page out. A new spill is refused when it would
// leave less than a tenth of the RAM (at least 4 GiB) available: the driver's own allocation is tried instead (alloc_driver).
void ram_info(unsigned long long& total, unsigned long long& available) {
#ifdef _WIN32
    MEMORYSTATUSEX status = {};
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    total = status.ullTotalPhys;
    available = status.ullAvailPhys;
#else
    total = available = 0;
    FILE* meminfo = fopen("/proc/meminfo", "r");
    char line[256];
    while (meminfo && fgets(line, sizeof(line), meminfo)) {
        unsigned long long kb = 0;
        if (sscanf(line, "MemTotal: %llu kB", &kb) == 1) total = kb << 10;
        else if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) available = kb << 10;
    }
    if (meminfo) fclose(meminfo);
#endif
}

unsigned long long ram_margin(unsigned long long total) {
    return std::max(4ull << 30, total / 10);
}

bool ram_short(size_t bytes) {
    unsigned long long total, available;
    ram_info(total, available);
    return available < bytes + ram_margin(total);
}

void release_spill_cache(Device& d) {  // after a synchronization: unmapping the range frees its system RAM
    for (auto& entry : d.spill_cache) {
        cu.MemUnmap(entry.first, entry.second.size);
        cu.MemAddressFree(entry.first, entry.second.size);
        d.spilled_bytes -= static_cast<int64_t>(entry.second.size);
    }
    d.spill_cache.clear();
}

void* alloc_spilled(Device& d, size_t size, CUstream stream) {
    // Spill mode, VRAM has no room for the tensor: it goes to pinned system RAM mapped into the GPU's address space (read and written
    // over PCIe), explicitly rather than through the Windows driver's fallback, which can lose the CUDA context when it fails.
    if (!d.host_granularity) return nullptr;
    size_t mapped = (size + d.host_granularity - 1) / d.host_granularity * d.host_granularity;
    for (auto it = d.spill_cache.begin(); it != d.spill_cache.end(); ++it) {  // a freed spilled tensor of the same size: no new pinning
        if (it->second.size == mapped && reusable(it->second.freed, it->second.stream, stream)) {
            CUdeviceptr va = it->first;
            d.live_spilled[va] = {mapped, stream, nullptr};
            d.spill_cache.erase(it);
            return reinterpret_cast<void*>(va);
        }
    }
    CUmemAllocationProp host = d.prop;
    host.location = {CU_MEM_LOCATION_TYPE_HOST, 0};
    CUmemGenericAllocationHandle handle;
    CUresult result = ram_short(mapped) ? CUDA_ERROR_OUT_OF_MEMORY : cu.MemCreate(&handle, mapped, &host, 0);
    if (result != CUDA_SUCCESS && !d.spill_cache.empty() && d.captures.empty()) {  // the cached spills of other sizes give their RAM back
        cu.CtxSynchronize();
        release_spill_cache(d);
        result = ram_short(mapped) ? CUDA_ERROR_OUT_OF_MEMORY : cu.MemCreate(&handle, mapped, &host, 0);
    }
    if (result != CUDA_SUCCESS) return nullptr;
    CUdeviceptr va = 0;
    if (cu.MemAddressReserve(&va, mapped, 0, 0, 0) != CUDA_SUCCESS) {
        cu.MemRelease(handle);
        return nullptr;
    }
    cu.MemMap(va, mapped, 0, handle, 0);
    cu.MemRelease(handle);  // the memory is freed when the range is unmapped
    cu.MemSetAccess(va, mapped, &d.access, 1);
    d.live_spilled[va] = {mapped, stream, nullptr};
    d.spilled_bytes += static_cast<int64_t>(mapped);
    return reinterpret_cast<void*>(va);
}

void* alloc_driver(Device& d, size_t size, CUstream stream) {
    // Spill mode, last resort (RAM too short to pin a spill): the driver's own allocation, as PyTorch's allocator makes it. With Windows'
    // driver model (also under WSL) it may go beyond the VRAM, to shared GPU memory, which pageable system RAM backs: what PyTorch's
    // allocator would have done. The memory this allocator keeps free first goes back to the driver, which may then place it in VRAM.
    if (d.captures.empty()) {
        cu.CtxSynchronize();
        release_cached_memory(d);
    }
    CUdeviceptr dptr = 0;
    if (cu.MemAlloc(&dptr, size) != CUDA_SUCCESS) return nullptr;
    d.live_spilled[dptr] = {size, stream, nullptr, true};
    d.driver_spilled += static_cast<int64_t>(size);
    return reinterpret_cast<void*>(dptr);
}

void take_chunks(Device& d, Range& range, size_t need, CUstream stream, bool wait_for_others) {
    // free chunks first, then the chunks of the oldest freed ranges, whatever their size; memory freed on another stream is taken
    // once that stream has finished with it, unless VRAM has run out
    for (size_t i = 0; i < d.spare.size() && range.chunks.size() < need;) {
        if (wait_for_others || reusable(d.spare[i].freed, d.spare[i].stream, stream)) {
            if (d.spare[i].stream != stream) wait(stream, d.spare[i].freed);
            range.chunks.push_back(d.spare[i].handle);
            d.spare[i] = d.spare.back();
            d.spare.pop_back();
        } else {
            ++i;
        }
    }
    for (auto it = d.cached.begin(); it != d.cached.end() && range.chunks.size() < need;) {
        if (wait_for_others || reusable(it->freed, it->stream, stream)) {
            Range old = std::move(*it);
            it = d.cached.erase(it);
            if (old.stream != stream) wait(stream, old.freed);
            for (auto handle : old.chunks) {
                if (range.chunks.size() < need) range.chunks.push_back(handle);
                else d.spare.push_back({handle, old.freed, old.stream});
            }
            retire({old.va, old.size, old.freed, d.context});
        } else {
            ++it;
        }
    }
}

void trim_driver_pools(Device& d) {
    // Before new chunks, the driver pools return the memory they keep free. They only release frees that a synchronization has observed:
    // the host synchronizes with the newest large free that has already completed, which returns at once, so what the pools freed on that
    // stream before it (an earlier phase or stage) is released without waiting for the GPU.
    unsigned long long pools_free = 0;
    for (CUmemoryPool pool : {d.small_pool, d.mid_pool})
        pools_free += pool_attribute(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT) - pool_attribute(pool, CU_MEMPOOL_ATTR_USED_MEM_CURRENT);
    if (pools_free < chunk_size || !d.captures.empty()) return;
    for (size_t i = d.probes.size(); i-- > 0;) {
        if (cu.EventQuery(d.probes[i].get()) == CUDA_SUCCESS) {
            cu.EventSynchronize(d.probes[i].get());
            trim_pools(d);
            d.probes.erase(d.probes.begin(), d.probes.begin() + static_cast<std::ptrdiff_t>(i));  // older ones observe nothing more
            return;
        }
    }
}

void* alloc_large(Device& d, size_t size, CUstream stream) {
    size_t need = (size + chunk_size - 1) / chunk_size;
    for (auto it = d.cached.begin(); it != d.cached.end(); ++it) {  // a freed range of the same size is reused as it is
        if (it->chunks.size() == need && reusable(it->freed, it->stream, stream)) {
            Range range = std::move(*it);
            d.cached.erase(it);
            range.stream = stream;
            range.freed.reset();
            CUdeviceptr va = range.va;
            d.live.emplace(va, std::move(range));
            return reinterpret_cast<void*>(va);
        }
    }
    Range range;
    range.size = need * chunk_size;
    range.stream = stream;
    take_chunks(d, range, need, stream, false);
    if (range.chunks.size() < need) trim_driver_pools(d);
    while (range.chunks.size() < need) {  // then new chunks
        CUmemGenericAllocationHandle handle;
        CUresult result = create_chunk(d, &handle, need - range.chunks.size());
        if (result == CUDA_ERROR_OUT_OF_MEMORY) {  // VRAM has run out: memory still used by other streams is taken after them
            ++d.recoveries;
            take_chunks(d, range, need, stream, true);
            if (range.chunks.size() == need) break;
            if (d.captures.empty()) {  // the driver pools and the small cache may hold memory no longer used
                cu.CtxSynchronize();
                free_small_cache(d);
                trim_pools(d);
                result = create_chunk(d, &handle, need - range.chunks.size());
            }
        }
        if (result != CUDA_SUCCESS) {
            for (auto h : range.chunks) d.spare.push_back({h, nullptr, stream});
            return nullptr;
        }
        range.chunks.push_back(handle);
        ++d.chunks_total;
    }
    if (cu.MemAddressReserve(&range.va, range.size, chunk_size, 0, 0) != CUDA_SUCCESS) {
        for (auto h : range.chunks) d.spare.push_back({h, nullptr, stream});
        return nullptr;
    }
    for (size_t i = 0; i < need; ++i) cu.MemMap(range.va + i * chunk_size, chunk_size, 0, range.chunks[i], 0);
    cu.MemSetAccess(range.va, range.size, &d.access, 1);
    d.peak_reserved = std::max(d.peak_reserved, reserved_now(d));
    CUdeviceptr va = range.va;
    d.live.emplace(va, std::move(range));
    return reinterpret_cast<void*>(va);
}

void* alloc_small(Device& d, size_t size, CUstream stream) {
    size = small_size(size);
    if (size <= small_cache_max) {
        auto lists = d.small_cache.find(stream);
        if (lists != d.small_cache.end()) {
            auto sized = lists->second.find(size);
            if (sized != lists->second.end() && !sized->second.empty()) {  // freed earlier on this stream: no ordering needed
                CUdeviceptr dptr = sized->second.back();
                sized->second.pop_back();
                d.small_cached -= static_cast<int64_t>(size);
                d.live_small[dptr] = {size, stream};
                return reinterpret_cast<void*>(dptr);
            }
        }
    }
    CUdeviceptr dptr = 0;
    CUmemoryPool pool = size >= mid_threshold ? d.mid_pool : d.small_pool;
    CUresult result = pool_alloc(d, &dptr, size, pool, stream);
    if (result == CUDA_ERROR_OUT_OF_MEMORY && d.captures.empty()) {  // freed large ranges, spare chunks and cached blocks give their memory back
        ++d.recoveries;
        cu.CtxSynchronize();
        release_cached_memory(d);
        result = pool_alloc(d, &dptr, size, pool, stream);
    }
    if (result != CUDA_SUCCESS) return nullptr;
    d.live_small[dptr] = {size, stream};
    if ((++d.small_calls & 255) == 0) d.peak_reserved = std::max(d.peak_reserved, reserved_now(d));
    return reinterpret_cast<void*>(dptr);
}

void* alloc_pool(Device& d, const MempoolId& id, size_t size) {
    Pool& pool = d.pools[id];
    size = (size + 511) & ~size_t(511);
    CUdeviceptr dptr = 0;
    auto fit = pool.free_blocks.lower_bound(size);
    if (fit != pool.free_blocks.end() && fit->first <= size + size / 4 + (1ull << 20)) {  // a free block of the pool that fits without much waste
        dptr = fit->second;
        pool.free_blocks.erase(fit);
    } else {
        if (cu.MemAlloc(&dptr, size) != CUDA_SUCCESS) return nullptr;
        pool.blocks[dptr] = size;
        d.pools_bytes += static_cast<int64_t>(size);
        d.peak_reserved = std::max(d.peak_reserved, reserved_now(d));
    }
    ++pool.live;
    d.pool_of[dptr] = id;
    return reinterpret_cast<void*>(dptr);
}

void release_pool_memory(Device& d, std::map<MempoolId, Pool>::iterator pool) {  // no graph uses the pool and none of its memory is live
    for (auto& block : pool->second.blocks) {
        cu.MemFree(block.first);
        d.pools_bytes -= static_cast<int64_t>(block.second);
    }
    d.pools.erase(pool);
}

void free_pool(Device& d, CUdeviceptr dptr, const MempoolId& id) {
    auto pool = d.pools.find(id);
    pool->second.free_blocks.emplace(pool->second.blocks[dptr], dptr);
    if (--pool->second.live == 0 && pool->second.use_count == 0) release_pool_memory(d, pool);
}

void release(Device& d, CUdeviceptr dptr) {
    auto it = d.live.find(dptr);
    auto small = it == d.live.end() ? d.live_small.find(dptr) : d.live_small.end();
    auto spilled = it == d.live.end() && small == d.live_small.end() ? d.live_spilled.find(dptr) : d.live_spilled.end();
    if (spilled != d.live_spilled.end() && spilled->second.driver) {  // cuMemFree waits for the work using it, on every stream
        if (!d.captures.empty()) {  // a synchronization would invalidate the captures: freed once they end
            d.deferred.push_back(dptr);
            return;
        }
        d.stream_uses.erase(dptr);
        d.driver_spilled -= static_cast<int64_t>(spilled->second.size);
        d.live_spilled.erase(spilled);
        cu.MemFree(dptr);
        return;
    }
    CUstream stream = it != d.live.end() ? it->second.stream : small != d.live_small.end() ? small->second.stream : spilled->second.stream;
    auto uses = d.stream_uses.find(dptr);
    if (uses != d.stream_uses.end()) {  // reuse must also wait for the other streams that used the memory
        for (CUstream other : uses->second) wait(stream, record_event(other));
        d.stream_uses.erase(uses);
    }
    if (it != d.live.end()) {
        Range range = std::move(it->second);
        d.live.erase(it);
        range.freed = record_event(range.stream);
        d.probes.push_back(range.freed);
        if (d.probes.size() > 32) d.probes.pop_front();
        d.cached.push_back(std::move(range));
        return;
    }
    if (spilled != d.live_spilled.end()) {
        Spilled entry = spilled->second;
        d.live_spilled.erase(spilled);
        entry.freed = record_event(stream);
        d.spill_cache.emplace_back(dptr, std::move(entry));
        return;
    }
    size_t size = small->second.size;
    d.live_small.erase(small);
    if (size <= small_cache_max && d.small_cached + static_cast<int64_t>(size) <= static_cast<int64_t>(small_cache_limit)) {
        d.small_cache[stream][size].push_back(dptr);
        d.small_cached += static_cast<int64_t>(size);
    } else {
        cu.MemFreeAsync(dptr, stream);
    }
}

const char* format_size(double bytes, char* text) {  // as PyTorch's out of memory message
    if (bytes <= 1024) snprintf(text, 32, "%.0f bytes", bytes);
    else if (bytes <= 1048576) snprintf(text, 32, "%.2f KiB", bytes / 1024);
    else if (bytes <= 1073741824) snprintf(text, 32, "%.2f MiB", bytes / 1048576);
    else snprintf(text, 32, "%.2f GiB", bytes / 1073741824);
    return text;
}

[[noreturn]] void out_of_memory(const Device& d, size_t size, int device) {
    size_t free_bytes = 0, total = 0;
    cu.MemGetInfo(&free_bytes, &total);
    NvmlMemory gpu = {};  // the whole GPU's free memory, read now: the driver's figure is this process' budget on Windows
    bool whole_gpu = d.nvml_device && nvml.DeviceGetMemoryInfo(d.nvml_device, &gpu) == 0;
    char request[32], capacity[32], available[32], driver_free[32], allocated[32], unused[32], need[32], headroom[32], spilled[32], ram[32],
        margin[32], message[1280];
    int length = snprintf(message, sizeof(message), "CUDA out of memory. Tried to allocate %s. GPU %d has a total capacity of %s of which %s is free",
             format_size(static_cast<double>(size), request), device, format_size(static_cast<double>(total), capacity),
             format_size(static_cast<double>(whole_gpu ? gpu.free : free_bytes), available));
    if (whole_gpu)
        length += snprintf(message + length, sizeof(message) - length, " (the driver reports %s free for this process)",
                           format_size(static_cast<double>(free_bytes), driver_free));
    length += snprintf(message + length, sizeof(message) - length, ". Of the allocated memory %s is allocated by PyTorch, and %s is reserved by "
             "the MMGP VRAM allocator but unallocated.", format_size(static_cast<double>(d.allocated), allocated),
             format_size(static_cast<double>(std::max<int64_t>(reserved_now(d) - d.allocated, 0)), unused));
    static const char* reasons[] = {"", "no VRAM was left for this process", "other programs use the rest of the GPU's VRAM",
                                    "the VRAM limit set for this process was reached", "the CUDA driver refused it"};
    if (d.refusal)
        length += snprintf(message + length, sizeof(message) - length, " %s more VRAM was needed (%s of it kept free for the driver), refused "
                           "because %s.", format_size(static_cast<double>(d.refused_need), need),
                           format_size(static_cast<double>(d.refused_headroom), headroom), reasons[d.refusal]);
    if (spill) {  // the allocation could not spill either
        unsigned long long ram_total, ram_available;
        ram_info(ram_total, ram_available);
        length += snprintf(message + length, sizeof(message) - length, " Spilling into system RAM (%s spilled) stopped: %s of RAM available, %s kept "
                           "free for the system.", format_size(static_cast<double>(d.spilled_bytes), spilled), format_size(static_cast<double>(ram_available), ram),
                           format_size(static_cast<double>(ram_margin(ram_total)), margin));
    }
    if (driver_spill && d.refusal != REFUSED_BY_LIMIT) {
#ifdef _WIN32
        snprintf(message + length, sizeof(message) - length, " The CUDA driver could not place it in shared GPU memory either (NVIDIA Control Panel: "
                 "CUDA - Sysmem Fallback Policy).");
#else
        snprintf(message + length, sizeof(message) - length, " The CUDA driver could not allocate it either.");
#endif
    }
    if (oom_thrower) oom_thrower(message);  // PyTorch's own error, which cuDNN and Python catch as such
    throw std::runtime_error(message);
}

double debug_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - debug_t0).count();
}

void debug_snapshot(Device& d, DebugSnapshot& snapshot, int64_t request) {
    int64_t cached = 0;
    for (auto& range : d.cached) cached += static_cast<int64_t>(range.size);
    snapshot.header = {debug_phase, d.allocated, reserved_now(d), d.spilled_bytes, d.chunks_total * static_cast<int64_t>(chunk_size),
                       static_cast<int64_t>(pool_attribute(d.small_pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)),
                       static_cast<int64_t>(pool_attribute(d.mid_pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)), d.pools_bytes, cached,
                       static_cast<int64_t>(d.spare.size() * chunk_size), d.small_cached, request, static_cast<int64_t>(d.debug_live.size()), debug_now()};
    snapshot.entries.clear();
    snapshot.entries.reserve(d.debug_live.size());
    for (auto& entry : d.debug_live) snapshot.entries.push_back(entry.second);
}

void debug_flush(Device& d) {  // the memory in use is still at the new maximum: copied now
    if (!d.debug_pending) return;
    debug_snapshot(d, d.debug_peaks[debug_phase], 0);
    d.debug_pending = false;
}

void debug_allocated(Device& d, void* ptr, size_t size, int64_t origin, int64_t kind, size_t threshold) {
    if (size >= threshold) {
        CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
        d.debug_live[dptr] = {dptr, static_cast<int64_t>(size), origin, ++debug_seq, kind, debug_now()};
        OriginTotals& totals = d.debug_origins[origin];
        ++totals.count;
        totals.bytes += static_cast<int64_t>(size);
        totals.max_size = std::max(totals.max_size, static_cast<int64_t>(size));
    }
    DebugSnapshot& peak = d.debug_peaks[debug_phase];
    if (d.allocated > peak.header.allocated) {
        peak.header.allocated = d.allocated;
        d.debug_pending = true;
    }
}

void debug_freed(Device& d, CUdeviceptr dptr) {
    debug_flush(d);
    auto entry = d.debug_live.find(dptr);
    if (entry == d.debug_live.end()) return;
    OriginTotals& totals = d.debug_origins[entry->second.origin];
    ++totals.freed;
    totals.life_us += static_cast<int64_t>((debug_now() - entry->second.time) * 1e6);
    d.debug_live.erase(entry);
}

void debug_clear(Device& d, bool keep_live) {
    if (!keep_live) d.debug_live.clear();
    d.debug_peaks.clear();
    d.debug_oom = {};
    d.debug_pending = false;
    d.debug_origins.clear();
}

int64_t copy_entries(const std::vector<DebugEntry>& source, DebugEntry* out, int64_t max) {
    for (int64_t i = 0; i < std::min<int64_t>(max, static_cast<int64_t>(source.size())); ++i) out[i] = source[i];
    return static_cast<int64_t>(source.size());
}

void end_capture(Device& d, std::vector<Capture>::iterator capture) {
    d.captures.erase(capture);
    if (d.captures.empty()) {  // the frees postponed during the captures
        std::vector<CUdeviceptr> pending;
        pending.swap(d.deferred);
        for (auto dptr : pending) release(d, dptr);
    }
}
}  // namespace

EXPORT int vmm_check(int device) {  // 0 when this device can use the allocator; else the reason (see mmgp/allocator)
    std::lock_guard<std::mutex> guard(lock);
    int status = load_driver();
    if (status) return status;
    CUdevice dev;
    if (cu.DeviceGet(&dev, device) != CUDA_SUCCESS) return 4;
    int vmm = 0, pools = 0;
    cu.DeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev);
    cu.DeviceGetAttribute(&pools, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, dev);
    return vmm ? (pools ? 0 : 6) : 5;
}

EXPORT void* vmm_alloc(size_t size, int device, CUstream stream) {
    if (size == 0) return nullptr;
    size_t threshold = debug_threshold.load(std::memory_order_relaxed);
    int64_t origin = -1;
    if (threshold && size >= threshold && !in_debug_origin) {  // debug mode: asked to Python before taking the lock, as it takes the GIL
        in_debug_origin = true;
        origin = debug_origin(size, device);
        in_debug_origin = false;
    }
    std::unique_lock<std::mutex> guard(lock);
    Device& d = device_state(device);
    RelaxedCapture relaxed(any_capture());
    if (!in_pressure_callback) d.refusal = 0;
    void* ptr = nullptr;
    bool to_pool = false;
    for (auto capture = d.captures.rbegin(); capture != d.captures.rend() && !to_pool; ++capture) {
        if (capture->filter(stream)) {
            ptr = alloc_pool(d, capture->id, size);
            to_pool = true;
        }
    }
    if (!to_pool) ptr = size >= large_threshold ? alloc_large(d, size, stream) : alloc_small(d, size, stream);
    if (!ptr && !to_pool && pressure_callback && !in_pressure_callback && d.captures.empty()) {
        // the application may free memory it can do without (model blocks it keeps in VRAM) before the allocation spills or fails: called
        // without the lock, as its frees come back here, then the allocation is tried again
        auto callback = pressure_callback;
        guard.unlock();
        in_pressure_callback = true;
        int64_t freed = callback(size, device);
        in_pressure_callback = false;
        guard.lock();
        if (freed > 0) ptr = size >= large_threshold ? alloc_large(d, size, stream) : alloc_small(d, size, stream);
    }
    int64_t kind = to_pool ? 4 : size >= large_threshold ? 0 : size >= mid_threshold ? 2 : 1;
    if (!ptr && !to_pool && d.refusal == REFUSED_BY_GPU && d.captures.empty()) {
        // before spilling, as PyTorch's allocator: this process' budget has room, the VRAM missing is used by other processes
        d.beyond_gpu = true;
        ptr = size >= large_threshold ? alloc_large(d, size, stream) : alloc_small(d, size, stream);
        d.beyond_gpu = false;
    }
    if (!ptr && spill && !to_pool) {
        ptr = alloc_spilled(d, size, stream);
        if (ptr) kind = 3;
    }
    if (!ptr && driver_spill && !to_pool && d.refusal != REFUSED_BY_LIMIT) {  // the VRAM limit emulates a smaller GPU: it stays hard
        ptr = alloc_driver(d, size, stream);
        if (ptr) kind = 5;
    }
    if (!ptr) {
        if (threshold) {
            debug_flush(d);
            debug_snapshot(d, d.debug_oom, static_cast<int64_t>(size));
        }
        out_of_memory(d, size, device);
    }
    count(d, static_cast<int64_t>(size));
    if (threshold) debug_allocated(d, ptr, size, origin, kind, threshold);
    return ptr;
}

EXPORT void vmm_free(void* ptr, size_t size, int device, CUstream stream) {
    if (!ptr) return;
    std::lock_guard<std::mutex> guard(lock);
    Device& d = devices.at(device);
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
    if (debug_threshold.load(std::memory_order_relaxed)) debug_freed(d, dptr);
    count(d, -static_cast<int64_t>(size));
    RelaxedCapture relaxed(any_capture());
    auto pooled = d.pool_of.find(dptr);
    if (pooled != d.pool_of.end()) {  // graph memory: replays follow the captured order, no event is needed
        MempoolId id = pooled->second;
        d.pool_of.erase(pooled);
        free_pool(d, dptr, id);
        return;
    }
    if (!d.captures.empty()) {  // no event may be recorded on a capturing stream
        bool any_capturing = capturing(stream);
        auto uses = d.stream_uses.find(dptr);
        if (uses != d.stream_uses.end())
            for (CUstream other : uses->second) any_capturing = any_capturing || capturing(other);
        if (any_capturing) {
            d.deferred.push_back(dptr);
            return;
        }
    }
    release(d, dptr);
}

EXPORT void vmm_record_stream(void* ptr, CUstream stream) {
    std::lock_guard<std::mutex> guard(lock);
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
    for (auto& entry : devices) {
        Device& d = entry.second;
        if (d.pool_of.count(dptr)) return;
        auto it = d.live.find(dptr);
        auto small = d.live_small.find(dptr);
        auto spilled = d.live_spilled.find(dptr);
        if (it == d.live.end() && small == d.live_small.end() && spilled == d.live_spilled.end()) continue;
        CUstream own = it != d.live.end() ? it->second.stream : small != d.live_small.end() ? small->second.stream : spilled->second.stream;
        auto& uses = d.stream_uses[dptr];
        if (stream != own && std::find(uses.begin(), uses.end(), stream) == uses.end()) uses.push_back(stream);
        return;
    }
}

EXPORT void vmm_begin_allocate_to_pool(int device, MempoolId id, CaptureFilter filter) {
    std::lock_guard<std::mutex> guard(lock);
    Device& d = device_state(device);
    ++d.pools[id].use_count;
    d.captures.push_back({id, std::move(filter)});
}

EXPORT void vmm_end_allocate_to_pool(int device, MempoolId id) {
    std::lock_guard<std::mutex> guard(lock);
    Device& d = device_state(device);
    for (auto capture = d.captures.rbegin(); capture != d.captures.rend(); ++capture) {
        if (capture->id == id) {
            end_capture(d, std::next(capture).base());
            return;
        }
    }
}

EXPORT void vmm_release_pool(int device, MempoolId id) {
    std::lock_guard<std::mutex> guard(lock);
    Device& d = device_state(device);
    auto pool = d.pools.find(id);
    if (pool == d.pools.end()) return;
    if (--pool->second.use_count <= 0 && pool->second.live == 0) {
        RelaxedCapture relaxed(any_capture());
        release_pool_memory(d, pool);
    }
}

EXPORT void vmm_configure(size_t large, size_t chunk, int spill_modes) {  // spill_modes: 1 pinned system RAM, 2 the driver's allocation
    std::lock_guard<std::mutex> guard(lock);
    large_threshold = large;
    chunk_size = chunk;
    spill = (spill_modes & 1) != 0;
    driver_spill = (spill_modes & 2) != 0;
}

EXPORT void vmm_set_vram_limit(int64_t bytes) {  // the VRAM this process may reserve on each device, 0 for no limit but the GPU's
    std::lock_guard<std::mutex> guard(lock);
    vram_limit = bytes;
}

EXPORT void vmm_stats(int device, int64_t* out) {  // allocated, reserved, peak allocated, peak reserved, chunks, cached ranges, stale ranges, live large, small cached, graph pools, spilled, recoveries, driver spilled
    std::lock_guard<std::mutex> guard(lock);
    for (int i = 0; i < 13; ++i) out[i] = 0;
    auto found = devices.find(device);
    if (found == devices.end()) return;
    Device& d = found->second;
    RelaxedCapture relaxed(any_capture());  // queried by hooks during captures: the pool query would invalidate them
    out[0] = d.allocated;
    out[1] = reserved_now(d);
    out[2] = d.peak_allocated;
    out[3] = std::max(d.peak_reserved, out[1]);
    out[4] = d.chunks_total;
    out[5] = static_cast<int64_t>(d.cached.size());
    {
        std::lock_guard<std::mutex> retired_guard(retired_lock);
        out[6] = static_cast<int64_t>(retired.size());
    }
    out[7] = static_cast<int64_t>(d.live.size());
    out[8] = d.small_cached;
    out[9] = d.pools_bytes;
    out[10] = d.spilled_bytes;
    out[11] = d.recoveries;
    out[12] = d.driver_spilled;
}

EXPORT void vmm_reset_peaks(int device) {
    std::lock_guard<std::mutex> guard(lock);
    auto found = devices.find(device);
    if (found == devices.end()) return;
    RelaxedCapture relaxed(any_capture());
    found->second.peak_allocated = found->second.allocated;
    found->second.peak_reserved = reserved_now(found->second);
}

EXPORT void vmm_set_pressure_callback(int64_t (*callback)(size_t, int)) {  // callback(size, device) -> bytes freed (see vmm_alloc); nullptr removes it
    std::lock_guard<std::mutex> guard(lock);
    pressure_callback = callback;
}

EXPORT void vmm_set_oom_thrower(void (*thrower)(const char*)) {  // throws c10::OutOfMemoryError(message), see oom_error.cpp; nullptr removes it
    std::lock_guard<std::mutex> guard(lock);
    oom_thrower = thrower;
}

EXPORT void vmm_mark_reset(int device) {  // starts a phase whose peak of allocated memory vmm_mark_peak returns
    std::lock_guard<std::mutex> guard(lock);
    auto found = devices.find(device);
    if (found != devices.end()) found->second.mark_peak = found->second.allocated;
}

EXPORT int64_t vmm_mark_peak(int device) {
    std::lock_guard<std::mutex> guard(lock);
    auto found = devices.find(device);
    return found == devices.end() ? 0 : found->second.mark_peak;
}

EXPORT int64_t vmm_room(int device) {  // VRAM that new memory may still take (see vram_short), the headroom not deducted
    std::lock_guard<std::mutex> guard(lock);
    auto found = devices.find(device);
    if (found == devices.end()) return 0;
    RelaxedCapture relaxed(any_capture());
    return room_now(found->second);
}

EXPORT void vmm_empty_cache() {  // on every device: the freed ranges, spare chunks, cached small blocks and freed spills are given back
    std::lock_guard<std::mutex> guard(lock);
    if (any_capture()) return;
    CUcontext current = nullptr;
    cu.CtxGetCurrent(&current);
    for (auto& entry : devices) {
        if (entry.second.context != current) cu.CtxSetCurrent(entry.second.context);
        cu.CtxSynchronize();
        release_cached_memory(entry.second);
        release_spill_cache(entry.second);
    }
    if (!devices.empty()) cu.CtxSetCurrent(current);
}

// ------------------------------------------------------------------ debug mode, driven by mmgp/allocator/debug.py
EXPORT void vmm_debug_start(size_t threshold, int64_t (*origin)(size_t, int)) {
    std::lock_guard<std::mutex> guard(lock);
    debug_threshold.store(0);
    for (auto& entry : devices) debug_clear(entry.second, false);
    debug_origin = origin;
    debug_phase = debug_seq = 0;
    debug_t0 = std::chrono::steady_clock::now();
    debug_threshold.store(threshold);
}

EXPORT void vmm_debug_stop() {
    std::lock_guard<std::mutex> guard(lock);
    debug_threshold.store(0);
    for (auto& entry : devices) debug_clear(entry.second, false);
}

EXPORT void vmm_debug_reset() {  // new peaks and totals; the live records stay
    std::lock_guard<std::mutex> guard(lock);
    for (auto& entry : devices) debug_clear(entry.second, true);
}

EXPORT void vmm_debug_mark(int64_t phase) {  // the allocations that follow count for this phase's peak
    std::lock_guard<std::mutex> guard(lock);
    for (auto& entry : devices) debug_flush(entry.second);
    debug_phase = phase;
}

EXPORT int64_t vmm_debug_seq(void* ptr) {  // sequence number of the recorded allocation at ptr, -1 when not recorded
    std::lock_guard<std::mutex> guard(lock);
    for (auto& entry : devices) {
        auto found = entry.second.debug_live.find(reinterpret_cast<CUdeviceptr>(ptr));
        if (found != entry.second.debug_live.end()) return found->second.seq;
    }
    return -1;
}

EXPORT int64_t vmm_debug_phases(int device, int64_t* phases, int64_t max) {  // the phases whose peak was copied
    std::lock_guard<std::mutex> guard(lock);
    auto found = devices.find(device);
    if (found == devices.end()) return 0;
    debug_flush(found->second);
    int64_t n = 0;
    for (auto& peak : found->second.debug_peaks)
        if (peak.second.header.time > 0 && n < max) phases[n++] = peak.first;
    return n;
}

EXPORT int64_t vmm_debug_snapshot(int device, int64_t phase, DebugHeader* header, DebugEntry* entries, int64_t max) {
    // the peak of a phase, or with phase -1 the state when an allocation last failed; returns its number of entries
    std::lock_guard<std::mutex> guard(lock);
    auto found = devices.find(device);
    if (found == devices.end()) return 0;
    Device& d = found->second;
    debug_flush(d);
    const DebugSnapshot* snapshot = &d.debug_oom;
    if (phase >= 0) {
        auto peak = d.debug_peaks.find(phase);
        if (peak == d.debug_peaks.end()) return 0;
        snapshot = &peak->second;
    }
    *header = snapshot->header;
    return copy_entries(snapshot->entries, entries, max);
}

EXPORT int64_t vmm_debug_live(int device, DebugEntry* entries, int64_t max) {  // the recorded allocations alive now
    std::lock_guard<std::mutex> guard(lock);
    auto found = devices.find(device);
    if (found == devices.end()) return 0;
    std::vector<DebugEntry> live;
    for (auto& entry : found->second.debug_live) live.push_back(entry.second);
    return copy_entries(live, entries, max);
}

EXPORT int64_t vmm_debug_origins(int device, int64_t* out, int64_t max) {  // 6 values per origin, see OriginTotals; returns the number of origins
    std::lock_guard<std::mutex> guard(lock);
    auto found = devices.find(device);
    if (found == devices.end()) return 0;
    int64_t n = 0;
    for (auto& entry : found->second.debug_origins) {
        if (n < max) {
            int64_t* row = out + 6 * n;
            row[0] = entry.first;
            row[1] = entry.second.count;
            row[2] = entry.second.bytes;
            row[3] = entry.second.max_size;
            row[4] = entry.second.freed;
            row[5] = entry.second.life_us;
        }
        ++n;
    }
    return n;
}
