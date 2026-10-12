// Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// MMGP RAM allocator and RAM debug hooks (ram.py, ram_debug.py), installed as PyTorch's CPU allocator.
// - Allocator: CPU tensors of min_size and more get their own pages in large reserved address ranges, committed at their size (rounded to
//   64 KiB, or 2 MiB for huge pages on Linux) instead of an arena block. Freed blocks are kept for reuse (a loop that allocates the same sizes
//   again reuses pages already in memory), up to cache_max, and go back to the system on demand (ra_release: the end of a generation, a
//   model release, the VRAM allocator before it spills into RAM) or when the workload needs it (a free beyond cache_max, new pages wanted
//   while the system's available RAM is short), in the thread that frees or allocates: no background thread, no timer, nothing runs while
//   the GPU works. PyTorch's own CPU allocator (mimalloc on Windows, glibc's arenas on Linux) keeps freed memory for good. Smaller tensors
//   keep PyTorch's allocator.
// - Debug hooks: the allocations of 64 KiB and more of PyTorch's CPU allocator and NumPy's default allocator are recorded with their origin
//   (from Python, for those of the threshold and more), the peak of each phase and the totals of each origin.
// Loaded only for the PyTorch versions listed in ram.py (TORCH_VERSIONS): c10::Allocator, DataPtr and SetAllocator are identical from 2.6 to
// the 2.15 development branch.
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sys/mman.h>
#include <cstdio>
#endif
#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
// see oom_error.cpp: the C++ runtime headers of recent toolchains read __libc_single_threaded (glibc 2.32+)
extern "C" {
__attribute__((visibility("hidden"))) char __libc_single_threaded = 0;
}
#endif

namespace {

// ------------------------------------------------------------------------------------------------------------------------------- allocator
constexpr size_t GRAIN = 64 << 10;         // block sizes and addresses: the allocation granularity of Windows
constexpr size_t HUGE_PAGE = 2 << 20;      // Linux: blocks of this size and more are aligned for transparent huge pages
constexpr size_t RANGE = size_t(64) << 30; // address space reserved at a time
constexpr int MAX_RANGES = 1024;

struct Block {
    uintptr_t addr;
    size_t size;
    double freed;
};

struct RamState {  // created by ra_install and never destroyed: frees may arrive until the process ends
    std::mutex mutex;
    std::map<uintptr_t, size_t> free_va;          // reserved address ranges with no committed pages, coalesced
    std::unordered_map<uintptr_t, size_t> live;   // blocks in use: address -> committed size
    std::vector<Block> cache;                     // freed blocks, still committed, kept for reuse
};

RamState* ram = nullptr;
std::atomic<bool> ram_enabled{false};
std::atomic<int> range_count{0};
uintptr_t range_base[MAX_RANGES], range_end[MAX_RANGES];  // published before range_count grows: ram_owns reads them without the lock
size_t min_size = 1 << 20, cache_max = size_t(2) << 30, pressure_bytes = size_t(4) << 30;
int64_t ram_live = 0, ram_cached = 0, ram_peak = 0, n_allocs = 0, n_hits = 0, n_fresh = 0, n_trims = 0, n_fallbacks = 0, n_released = 0, released_bytes = 0;
const auto ram_t0 = std::chrono::steady_clock::now();

double ram_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - ram_t0).count(); }

size_t round_up(size_t value, size_t unit) { return (value + unit - 1) / unit * unit; }

#ifdef _WIN32
uintptr_t vm_reserve(size_t size) { return reinterpret_cast<uintptr_t>(VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS)); }
bool vm_commit(uintptr_t addr, size_t size) { return VirtualAlloc(reinterpret_cast<void*>(addr), size, MEM_COMMIT, PAGE_READWRITE) != nullptr; }
void vm_decommit(uintptr_t addr, size_t size) { VirtualFree(reinterpret_cast<void*>(addr), size, MEM_DECOMMIT); }
size_t available_ram() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<size_t>(status.ullAvailPhys);
}
#else
uintptr_t vm_reserve(size_t size) {
    void* addr = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return addr == MAP_FAILED ? 0 : reinterpret_cast<uintptr_t>(addr);
}
bool vm_commit(uintptr_t addr, size_t size) {
    if (mprotect(reinterpret_cast<void*>(addr), size, PROT_READ | PROT_WRITE) != 0) return false;
    if (size >= HUGE_PAGE) madvise(reinterpret_cast<void*>(addr), size, MADV_HUGEPAGE);  // fewer page faults where the system allows it
    return true;
}
void vm_decommit(uintptr_t addr, size_t size) {
    madvise(reinterpret_cast<void*>(addr), size, MADV_DONTNEED);
    mprotect(reinterpret_cast<void*>(addr), size, PROT_NONE);
}
size_t available_ram() {
    FILE* file = std::fopen("/proc/meminfo", "r");
    if (file == nullptr) return SIZE_MAX;
    char line[256];
    size_t kib = SIZE_MAX;
    while (std::fgets(line, sizeof(line), file)) {
        if (std::sscanf(line, "MemAvailable: %zu kB", &kib) == 1) break;
    }
    std::fclose(file);
    return kib == SIZE_MAX ? kib : kib << 10;
}
#endif

bool ram_owns(const void* ptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    int count = range_count.load(std::memory_order_acquire);
    for (int index = 0; index < count; index++) {
        if (addr >= range_base[index] && addr < range_end[index]) return true;
    }
    return false;
}

void va_give(uintptr_t addr, size_t size) {  // with the lock: an address range back to the free ones, merged with its neighbours
    auto next = ram->free_va.lower_bound(addr);
    if (next != ram->free_va.end() && addr + size == next->first) {
        size += next->second;
        next = ram->free_va.erase(next);
    }
    if (next != ram->free_va.begin()) {
        auto previous = std::prev(next);
        if (previous->first + previous->second == addr) {
            previous->second += size;
            return;
        }
    }
    ram->free_va.emplace(addr, size);
}

uintptr_t va_take(size_t size, size_t align) {  // with the lock: the first free address range that holds size at align, or a new range
    for (int attempt = 0; attempt < 2; attempt++) {
        for (auto it = ram->free_va.begin(); it != ram->free_va.end(); ++it) {
            uintptr_t start = round_up(it->first, align), end = it->first + it->second;
            if (start + size > end) continue;
            uintptr_t first = it->first;
            ram->free_va.erase(it);
            if (start > first) ram->free_va.emplace(first, start - first);
            if (start + size < end) ram->free_va.emplace(start + size, end - start - size);
            return start;
        }
        int index = range_count.load(std::memory_order_relaxed);
        size_t length = std::max(RANGE, round_up(size + align, RANGE));
        uintptr_t base = index < MAX_RANGES ? vm_reserve(length) : 0;
        if (base == 0) return 0;
        range_base[index] = base;
        range_end[index] = base + length;
        range_count.store(index + 1, std::memory_order_release);
        ram->free_va.emplace(base, length);
    }
    return 0;
}

void release_blocks(std::unique_lock<std::mutex>& guard, std::vector<Block>& blocks) {  // with the lock, released while decommitting
    if (blocks.empty()) return;
    guard.unlock();
    for (auto& block : blocks) vm_decommit(block.addr, block.size);
    guard.lock();
    for (auto& block : blocks) {
        va_give(block.addr, block.size);
        n_released++;
        released_bytes += static_cast<int64_t>(block.size);
    }
    blocks.clear();
}

void take_cache(std::vector<Block>& out) {  // with the lock: every cached block
    for (auto& block : ram->cache) {
        out.push_back(block);
        ram_cached -= static_cast<int64_t>(block.size);
    }
    ram->cache.clear();
}

void* ram_alloc(size_t n) {
    size_t align = GRAIN;
#ifndef _WIN32
    if (n >= HUGE_PAGE) align = HUGE_PAGE;
#endif
    size_t size = round_up(n, align);
    std::unique_lock<std::mutex> guard(ram->mutex);
    n_allocs++;
    // the smallest cached block that holds it, at most twice its size (larger blocks stay for larger requests); the most recent first
    size_t best = SIZE_MAX;
    for (size_t index = 0; index < ram->cache.size(); index++) {
        const Block& block = ram->cache[index];
        if (block.size < size || block.size > 2 * size) continue;
        if (best == SIZE_MAX || block.size < ram->cache[best].size || (block.size == ram->cache[best].size && block.freed > ram->cache[best].freed)) best = index;
    }
    if (best != SIZE_MAX) {
        Block block = ram->cache[best];
        ram->cache[best] = ram->cache.back();
        ram->cache.pop_back();
        ram_cached -= static_cast<int64_t>(block.size);
        size_t tail = block.size - size;
        if (tail < (1 << 20)) {
            size = block.size;  // a small excess stays with the block
            tail = 0;
        }
        ram->live[block.addr] = size;
        ram_live += static_cast<int64_t>(size);
        n_hits++;
        if (tail) {  // the excess goes back to the system
            n_trims++;
            std::vector<Block> excess{{block.addr + size, tail, 0}};
            release_blocks(guard, excess);
        }
        return reinterpret_cast<void*>(block.addr);
    }
    if (!ram->cache.empty()) {  // new pages: when the system's available RAM is short, the cached blocks go back first
        guard.unlock();
        bool short_of_ram = available_ram() < pressure_bytes + size;
        guard.lock();
        if (short_of_ram) {
            std::vector<Block> cached;
            take_cache(cached);
            release_blocks(guard, cached);
        }
    }
    uintptr_t addr = va_take(size, align);
    guard.unlock();
    bool committed = addr != 0 && vm_commit(addr, size);
    guard.lock();
    if (!committed) {  // short of memory: the cache goes back to the system first
        std::vector<Block> cached;
        take_cache(cached);
        release_blocks(guard, cached);
        if (addr == 0) addr = va_take(size, align);
        committed = addr != 0 && vm_commit(addr, size);
        if (!committed) {
            if (addr != 0) va_give(addr, size);
            n_fallbacks++;
            return nullptr;  // PyTorch's allocator takes this request (and raises its own error if the memory is really short)
        }
    }
    ram->live[addr] = size;
    ram_live += static_cast<int64_t>(size);
    ram_peak = std::max(ram_peak, ram_live + ram_cached);
    n_fresh++;
    return reinterpret_cast<void*>(addr);
}

void ram_free(void* ptr) {
    std::unique_lock<std::mutex> guard(ram->mutex);
    auto found = ram->live.find(reinterpret_cast<uintptr_t>(ptr));
    Block block{found->first, found->second, ram_now()};
    ram->live.erase(found);
    ram_live -= static_cast<int64_t>(block.size);
    std::vector<Block> released;
    if (block.size > cache_max) {  // a block the cache cannot hold goes back at once, as PyTorch's allocator does with large blocks
        released.push_back(block);
    } else {  // the least recently freed blocks beyond cache_max go back
        ram->cache.push_back(block);
        ram_cached += static_cast<int64_t>(block.size);
        if (ram_cached > static_cast<int64_t>(cache_max)) {
            std::sort(ram->cache.begin(), ram->cache.end(), [](const Block& a, const Block& b) { return a.freed < b.freed; });
            size_t count = 0;
            while (count < ram->cache.size() && ram_cached > static_cast<int64_t>(cache_max)) ram_cached -= static_cast<int64_t>(ram->cache[count++].size);
            released.assign(ram->cache.begin(), ram->cache.begin() + count);
            ram->cache.erase(ram->cache.begin(), ram->cache.begin() + count);
        }
    }
    release_blocks(guard, released);
}

// ------------------------------------------------------------------------------------------------------------------------------- debug
struct Entry {  // mirrored by _HookEntry in ram_debug.py
    uint64_t ptr;
    int64_t size, origin, seq, kind;  // kind: 0 PyTorch, 1 NumPy
    double time;
};

struct Header {  // mirrored by _HookHeader
    int64_t phase, torch_live, numpy_live, entries;
    double time;
};

struct Phase {
    int64_t peak = 0, torch_live = 0, numpy_live = 0, version = -1;  // version: of the large allocations copied in entries
    double time = 0;
    std::vector<Entry> entries;  // the allocations of the threshold and more alive at the peak
};

struct Totals {
    int64_t count = 0, total = 0, largest = 0, freed = 0, life_us = 0;
};

using OriginFn = int64_t (*)(size_t, int);
using ThreadStateFn = void* (*)();

constexpr int GRANULE_SHIFT = 25;                  // 32 MiB, mimalloc's segment size
constexpr uint64_t GRANULES = 1ull << (47 - GRANULE_SHIFT);  // user address space of x86_64 / aarch64

std::mutex lock;
std::unordered_map<uint64_t, Entry> live;       // the recorded allocations alive
std::unordered_map<uint64_t, Entry> large;      // those of the threshold and more
std::unordered_map<int64_t, Phase> phases;
std::unordered_map<int64_t, Totals> totals;     // key: origin * 2 + kind
std::atomic<uint16_t>* granule_counts = nullptr;  // recorded allocations starting in each granule: frees elsewhere skip the lock
std::atomic<uint64_t>* torch_granules = nullptr;  // bitmap of the granules that served PyTorch allocations
std::atomic<bool> recording{false};
int64_t threshold = 16 << 20, track_min = 64 << 10, seq = 0, current_phase = 0, large_version = 0;
int64_t live_bytes[2] = {0, 0}, live_count[2] = {0, 0};
std::atomic<int64_t> allocations{0}, unwrapped{0};
OriginFn origin_fn = nullptr;
ThreadStateFn thread_state = nullptr;  // PyGILState_GetThisThreadState: origins are asked only from Python threads
std::chrono::steady_clock::time_point t0;
thread_local bool in_origin = false;

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

uint64_t granule(const void* ptr) { return (reinterpret_cast<uint64_t>(ptr) >> GRANULE_SHIFT) & (GRANULES - 1); }

void track(void* ptr, size_t size, int kind) {
    int64_t origin = -1;
    if (static_cast<int64_t>(size) >= threshold && origin_fn != nullptr && !in_origin) {
        if (thread_state != nullptr && thread_state() != nullptr) {  // a thread that Python knows: taking the GIL cannot wait on an OpenMP worker
            in_origin = true;
            origin = origin_fn(size, kind);
            in_origin = false;
        } else {
            origin = -2;  // no Python thread
        }
    }
    double time = now();
    std::lock_guard<std::mutex> guard(lock);
    if (!recording.load(std::memory_order_relaxed)) return;
    Entry entry{reinterpret_cast<uint64_t>(ptr), static_cast<int64_t>(size), origin, seq++, kind, time};
    auto inserted = live.emplace(entry.ptr, entry);
    if (!inserted.second) {  // an address freed without our knowledge (allocated before the hooks): replace it
        live_bytes[inserted.first->second.kind] -= inserted.first->second.size;
        live_count[inserted.first->second.kind]--;
        if (large.erase(entry.ptr)) large_version++;
        inserted.first->second = entry;
    } else {
        granule_counts[granule(ptr)].fetch_add(1, std::memory_order_relaxed);
    }
    live_bytes[kind] += entry.size;
    live_count[kind]++;
    if (entry.size >= threshold) {
        large[entry.ptr] = entry;
        large_version++;
        Totals& sums = totals[origin * 2 + kind];
        sums.count++;
        sums.total += entry.size;
        if (entry.size > sums.largest) sums.largest = entry.size;
    }
    Phase& phase = phases[current_phase];
    int64_t total = live_bytes[0] + live_bytes[1];
    if (total > phase.peak) {
        phase.peak = total;
        phase.torch_live = live_bytes[0];
        phase.numpy_live = live_bytes[1];
        phase.time = time;
        if (phase.version != large_version) {
            phase.entries.clear();
            for (auto& item : large) phase.entries.push_back(item.second);
            phase.version = large_version;
        }
    }
}

void untrack(void* ptr) {
    if (ptr == nullptr || granule_counts == nullptr || granule_counts[granule(ptr)].load(std::memory_order_relaxed) == 0) return;
    std::lock_guard<std::mutex> guard(lock);
    auto found = live.find(reinterpret_cast<uint64_t>(ptr));
    if (found == live.end()) return;
    Entry& entry = found->second;
    live_bytes[entry.kind] -= entry.size;
    live_count[entry.kind]--;
    granule_counts[granule(ptr)].fetch_sub(1, std::memory_order_relaxed);
    if (entry.size >= threshold) {
        large.erase(entry.ptr);
        large_version++;
        Totals& sums = totals[entry.origin * 2 + entry.kind];
        sums.freed++;
        sums.life_us += static_cast<int64_t>((now() - entry.time) * 1e6);
    }
    live.erase(found);
}

void recorded(void* ptr, size_t n) {  // a PyTorch allocation, for the debug mode
    if (!recording.load(std::memory_order_relaxed)) return;
    allocations.fetch_add(1, std::memory_order_relaxed);
    uint64_t index = granule(ptr);
    torch_granules[index >> 6].fetch_or(1ull << (index & 63), std::memory_order_relaxed);
    if (static_cast<int64_t>(n) >= track_min) track(ptr, n, 0);
}

// ------------------------------------------------------------------------------------------------------------------------------- PyTorch
c10::Allocator* inner = nullptr;
c10::DeleterFnPtr inner_deleter = nullptr;

void hooked_delete(void* ptr) {
    untrack(ptr);
    if (ram_owns(ptr)) {
        ram_free(ptr);
    } else {
        inner_deleter(ptr);
    }
}

struct HookAllocator final : c10::Allocator {
    c10::DataPtr allocate(size_t n) override {
        if (n >= min_size && ram_enabled.load(std::memory_order_relaxed)) {
            void* ptr = ram_alloc(n);
            if (ptr != nullptr) {
                recorded(ptr, n);
                return {ptr, ptr, &hooked_delete, c10::Device(c10::DeviceType::CPU)};
            }
        }
        c10::DataPtr data = inner->allocate(n);
        void* ptr = data.get();
        if (data.get_context() != ptr || data.get_deleter() != inner_deleter) {  // not a simple allocation: left as it is
            unwrapped.fetch_add(1, std::memory_order_relaxed);
            return data;
        }
        c10::Device device = data.device();
        data.release_context();
        if (ptr != nullptr) recorded(ptr, n);
        return {ptr, ptr, &hooked_delete, device};
    }
    c10::DeleterFnPtr raw_deleter() const override { return &hooked_delete; }
    void copy_data(void* dest, const void* src, std::size_t count) const override { default_copy_data(dest, src, count); }
};

HookAllocator* hook_allocator = new HookAllocator();  // never destroyed: PyTorch may allocate until the process ends
constexpr uint8_t PRIORITY = 200;

int install_torch() {
    if (inner != nullptr) return 0;
    c10::Allocator* current = c10::GetAllocator(c10::DeviceType::CPU);
    if (current == nullptr || current->raw_deleter() == nullptr) return 1;  // an allocator whose allocations carry contexts: not handled
    inner_deleter = current->raw_deleter();
    inner = current;
    c10::SetAllocator(c10::DeviceType::CPU, hook_allocator, PRIORITY);
    return 0;
}

// ------------------------------------------------------------------------------------------------------------------------------- NumPy
struct NumpyAllocator {  // PyDataMemAllocator (NumPy 1.22+, handler version 1)
    void* ctx;
    void* (*malloc)(void*, size_t);
    void* (*calloc)(void*, size_t, size_t);
    void* (*realloc)(void*, void*, size_t);
    void (*free)(void*, void*, size_t);
};

struct NumpyHandler {  // PyDataMem_Handler
    char name[127];
    uint8_t version;
    NumpyAllocator allocator;
};

NumpyHandler* numpy_handler = nullptr;
NumpyAllocator numpy_original;

void* numpy_malloc(void* ctx, size_t size) {
    void* ptr = numpy_original.malloc(ctx, size);
    if (ptr != nullptr && static_cast<int64_t>(size) >= track_min && recording.load(std::memory_order_relaxed)) track(ptr, size, 1);
    return ptr;
}

void* numpy_calloc(void* ctx, size_t count, size_t item) {
    void* ptr = numpy_original.calloc(ctx, count, item);
    if (ptr != nullptr && static_cast<int64_t>(count * item) >= track_min && recording.load(std::memory_order_relaxed)) track(ptr, count * item, 1);
    return ptr;
}

void* numpy_realloc(void* ctx, void* ptr, size_t size) {
    untrack(ptr);
    void* moved = numpy_original.realloc(ctx, ptr, size);
    if (moved != nullptr && static_cast<int64_t>(size) >= track_min && recording.load(std::memory_order_relaxed)) track(moved, size, 1);
    return moved;
}

void numpy_free(void* ctx, void* ptr, size_t size) {
    untrack(ptr);
    numpy_original.free(ctx, ptr, size);
}

}  // namespace

// ------------------------------------------------------------------------------------------------------------------------------- allocator API
EXPORT int ra_install(int64_t min_bytes, int64_t cache_bytes, int64_t pressure) {
    if (install_torch() != 0) return 1;
    if (ram == nullptr) {
        ram = new RamState();
        min_size = static_cast<size_t>(min_bytes);
        cache_max = static_cast<size_t>(cache_bytes);
        pressure_bytes = static_cast<size_t>(pressure);
#ifndef _WIN32
        // fork (multiprocessing's default on Linux): the lock is held across it, so that the child does not inherit it locked by a thread it
        // does not have
        pthread_atfork([] { ram->mutex.lock(); }, [] { ram->mutex.unlock(); }, [] { ram->mutex.unlock(); });
#endif
    }
    ram_enabled = true;
    return 0;
}

EXPORT void ra_set_enabled(int enabled) { ram_enabled = enabled != 0 && ram != nullptr; }

EXPORT void ra_release() {  // every cached block back to the system now
    if (ram == nullptr) return;
    std::unique_lock<std::mutex> guard(ram->mutex);
    std::vector<Block> cached;
    take_cache(cached);
    release_blocks(guard, cached);
}

EXPORT void ra_stats(int64_t* out) {  // enabled, live, cached, peak (live + cached), allocations, cache hits, fresh, trims, fallbacks, releases, released bytes, reserved
    if (ram == nullptr) {
        std::fill(out, out + 12, 0);
        return;
    }
    std::lock_guard<std::mutex> guard(ram->mutex);
    int64_t reserved = 0;
    for (int index = 0; index < range_count.load(); index++) reserved += static_cast<int64_t>(range_end[index] - range_base[index]);
    int64_t values[12] = {ram_enabled.load(), ram_live, ram_cached, ram_peak, n_allocs, n_hits, n_fresh, n_trims, n_fallbacks, n_released, released_bytes, reserved};
    std::copy(values, values + 12, out);
}

EXPORT void ra_reset_peak() {
    if (ram == nullptr) return;
    std::lock_guard<std::mutex> guard(ram->mutex);
    ram_peak = ram_live + ram_cached;
}

EXPORT int64_t ra_ranges(uint64_t* out, int64_t capacity) {  // (base, size) of the reserved address ranges
    int64_t count = range_count.load(std::memory_order_acquire);
    for (int64_t index = 0; index < count && index < capacity; index++) {
        out[2 * index] = range_base[index];
        out[2 * index + 1] = range_end[index] - range_base[index];
    }
    return count;
}

// ------------------------------------------------------------------------------------------------------------------------------- debug API
EXPORT void rh_start(int64_t threshold_bytes, int64_t track_min_bytes, OriginFn origin, ThreadStateFn python_thread_state) {
    std::lock_guard<std::mutex> guard(lock);
    if (granule_counts == nullptr) {  // never freed: frees may arrive at any time
        granule_counts = new std::atomic<uint16_t>[GRANULES]();
        torch_granules = new std::atomic<uint64_t>[GRANULES / 64]();
    }
    t0 = std::chrono::steady_clock::now();
    for (auto& item : live) granule_counts[granule(reinterpret_cast<void*>(item.first))].fetch_sub(1, std::memory_order_relaxed);
    live.clear();
    large.clear();
    large_version++;
    phases.clear();
    totals.clear();
    live_bytes[0] = live_bytes[1] = live_count[0] = live_count[1] = 0;
    threshold = threshold_bytes;
    track_min = track_min_bytes;
    origin_fn = origin;
    thread_state = python_thread_state;
    current_phase = 0;
    recording = true;
}

EXPORT void rh_stop() {  // the hooks stay installed and forward: the memory they handed out is freed by the original allocators
    std::lock_guard<std::mutex> guard(lock);
    recording = false;
    origin_fn = nullptr;
}

EXPORT int rh_install_torch() { return install_torch(); }

EXPORT int rh_patch_numpy(NumpyHandler* handler) {
    if (numpy_handler != nullptr) return 0;
    if (handler == nullptr || handler->version != 1) return 1;
    numpy_original = handler->allocator;
    numpy_handler = handler;
    handler->allocator.malloc = numpy_malloc;  // pointer-sized stores: a thread reads the original or the hook, both forward correctly
    handler->allocator.calloc = numpy_calloc;
    handler->allocator.realloc = numpy_realloc;
    handler->allocator.free = numpy_free;
    return 0;
}

EXPORT void rh_mark(int64_t phase) {
    std::lock_guard<std::mutex> guard(lock);
    current_phase = phase;
    Phase& entry = phases[phase];
    int64_t total = live_bytes[0] + live_bytes[1];
    if (total > entry.peak) {  // a phase starts with what is alive
        entry.peak = total;
        entry.torch_live = live_bytes[0];
        entry.numpy_live = live_bytes[1];
        entry.time = now();
        entry.entries.clear();
        for (auto& item : large) entry.entries.push_back(item.second);
        entry.version = large_version;
    }
}

EXPORT void rh_reset() {  // new peaks and totals; the allocations alive stay recorded
    std::lock_guard<std::mutex> guard(lock);
    phases.clear();
    totals.clear();
}

EXPORT void rh_stats(int64_t* out) {
    std::lock_guard<std::mutex> guard(lock);
    out[0] = live_bytes[0];
    out[1] = live_bytes[1];
    out[2] = live_count[0];
    out[3] = live_count[1];
    out[4] = unwrapped.load();
    out[5] = allocations.load();
    out[6] = numpy_handler != nullptr;
    out[7] = inner != nullptr;
}

EXPORT int64_t rh_live(Entry* out, int64_t capacity) {
    std::lock_guard<std::mutex> guard(lock);
    int64_t count = 0;
    for (auto& item : live) {
        if (count < capacity && out != nullptr) out[count] = item.second;
        count++;
    }
    return count;
}

EXPORT int64_t rh_phases(int64_t* out, int64_t capacity) {
    std::lock_guard<std::mutex> guard(lock);
    int64_t count = 0;
    for (auto& item : phases) {
        if (count < capacity) out[count] = item.first;
        count++;
    }
    return count;
}

EXPORT int64_t rh_phase(int64_t phase, Header* header, Entry* out, int64_t capacity) {
    std::lock_guard<std::mutex> guard(lock);
    auto found = phases.find(phase);
    if (found == phases.end()) return 0;
    Phase& entry = found->second;
    *header = Header{phase, entry.torch_live, entry.numpy_live, static_cast<int64_t>(entry.entries.size()), entry.time};
    for (int64_t index = 0; out != nullptr && index < capacity && index < static_cast<int64_t>(entry.entries.size()); index++) out[index] = entry.entries[index];
    return static_cast<int64_t>(entry.entries.size());
}

EXPORT int64_t rh_origins(int64_t* rows, int64_t capacity) {  // rows of 7: origin, kind, count, total, largest, freed, life_us
    std::lock_guard<std::mutex> guard(lock);
    int64_t count = 0;
    for (auto& item : totals) {
        if (count < capacity) {
            int64_t* row = rows + 7 * count;
            int64_t kind = ((item.first % 2) + 2) % 2;
            row[0] = (item.first - kind) / 2;
            row[1] = kind;
            row[2] = item.second.count;
            row[3] = item.second.total;
            row[4] = item.second.largest;
            row[5] = item.second.freed;
            row[6] = item.second.life_us;
        }
        count++;
    }
    return count;
}

EXPORT int64_t rh_blocks(uint64_t* out, int64_t capacity) {  // the granules (address >> 25) that served PyTorch allocations
    if (torch_granules == nullptr) return 0;
    int64_t count = 0;
    for (uint64_t word = 0; word < GRANULES / 64; word++) {
        uint64_t bits = torch_granules[word].load(std::memory_order_relaxed);
        while (bits) {
            int bit = 0;
            while (!(bits & (1ull << bit))) bit++;
            bits &= bits - 1;
            if (count < capacity && out != nullptr) out[count] = word * 64 + bit;
            count++;
        }
    }
    return count;
}

#ifdef _WIN32
// The blocks in use in the process heap (malloc's): count and bytes by size (under 1 KiB, 64 KiB, 1 MiB, 16 MiB, more) in classes[10], the
// largest blocks of 1 MiB and more as (address, size) pairs in largest, and their first HEAD bytes in heads (read while the heap is locked:
// no block can be freed meanwhile). The walk allocates nothing.
constexpr int HEAD = 32;  // HEAD_BYTES in ram_debug.py

EXPORT int64_t rh_heap_walk(int64_t* classes, uint64_t* largest, char* heads, int64_t capacity) {
    std::vector<std::pair<uint64_t, uint64_t>> top;  // (size, address), a min-heap of the largest blocks
    top.reserve(static_cast<size_t>(capacity) + 1);
    std::fill(classes, classes + 10, 0);
    HANDLE heap = GetProcessHeap();
    if (!HeapLock(heap)) return -1;
    int64_t entries = 0;
    PROCESS_HEAP_ENTRY entry{};
    while (HeapWalk(heap, &entry)) {
        entries++;
        if (!(entry.wFlags & PROCESS_HEAP_ENTRY_BUSY)) continue;
        uint64_t size = entry.cbData;
        int index = size < (1u << 10) ? 0 : size < (64u << 10) ? 1 : size < (1u << 20) ? 2 : size < (16u << 20) ? 3 : 4;
        classes[2 * index]++;
        classes[2 * index + 1] += static_cast<int64_t>(size);
        if (capacity > 0 && size >= (1u << 20) && (static_cast<int64_t>(top.size()) < capacity || size > top.front().first)) {
            if (static_cast<int64_t>(top.size()) == capacity) {
                std::pop_heap(top.begin(), top.end(), std::greater<>());
                top.pop_back();
            }
            top.emplace_back(size, reinterpret_cast<uint64_t>(entry.lpData));
            std::push_heap(top.begin(), top.end(), std::greater<>());
        }
    }
    std::sort(top.begin(), top.end(), std::greater<>());
    for (int64_t index = 0; index < capacity; index++) {
        bool used = index < static_cast<int64_t>(top.size());
        largest[2 * index] = used ? top[index].second : 0;
        largest[2 * index + 1] = used ? top[index].first : 0;
        if (used) std::memcpy(heads + HEAD * index, reinterpret_cast<const void*>(top[index].second), HEAD);  // blocks of 1 MiB and more
    }
    HeapUnlock(heap);
    return entries;
}
#endif
