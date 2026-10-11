# Briefly oversubscribes VRAM so Windows (WDDM) trims the idle allocations of every process, then releases it.
import ctypes, os, subprocess, sys, time

CHUNK, OVERSHOOT_CHUNKS, HOLD_S, GUARD_MB = 256 << 20, 8, 2.0, 1024

ps = r"(Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples | ? CookedValue -gt 1GB | % { $p=[int]($_.InstanceName -replace '^pid_(\d+)_.*','$1'); (Get-Process -Id $p -ea 0).ProcessName + ' ' + $p }"
busy = [l for l in subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.split("\n") if l.strip() and not l.startswith("dwm ")]
if busy and "--force" not in sys.argv:
    sys.exit(f"Refusing: {', '.join(sorted(set(busy)))} uses more than {GUARD_MB} MB of VRAM; trimming would push it to system RAM. Use --force to trim anyway.")

cuda = ctypes.WinDLL("nvcuda.dll")
def ck(r, what):
    if r: sys.exit(f"{what} failed with CUDA error {r}")
dev, ctx = ctypes.c_int(), ctypes.c_void_p()
free, total = ctypes.c_size_t(), ctypes.c_size_t()
ck(cuda.cuInit(0), "cuInit")
ck(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ck(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
ck(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
ptrs, overshoot = [], OVERSHOOT_CHUNKS
while True:
    ck(cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
    if free.value < 2 * CHUNK:
        if overshoot == 0: break
        overshoot -= 1
    p = ctypes.c_uint64()
    if cuda.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(CHUNK)): break
    ck(cuda.cuMemsetD8_v2(p, ctypes.c_ubyte(0), ctypes.c_size_t(CHUNK)), "cuMemsetD8")
    ptrs.append(p)
ck(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
time.sleep(HOLD_S)
for p in ptrs: cuda.cuMemFree_v2(p)
cuda.cuDevicePrimaryCtxRelease(dev)
print(f"Held {len(ptrs) * CHUNK >> 20} MiB for {HOLD_S:.0f}s, released.")
