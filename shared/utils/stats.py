import gradio as gr
import signal
import sys
import time
import threading
import atexit
from contextlib import contextmanager
from collections import deque
from pathlib import Path
import psutil


class GpuBackend:
    """Base class for GPU stat providers."""
    def query(self):
        """Returns (gpu_percent, vram_used_bytes, vram_total_bytes)."""
        raise NotImplementedError


class NvmlBackend(GpuBackend):
    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def query(self):
        util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        return util.gpu, mem.used, mem.total


class AmdsmiBackend(GpuBackend):
    def __init__(self):
        import amdsmi
        amdsmi.amdsmi_init()
        devices = amdsmi.amdsmi_get_processor_handles()
        if not devices:
            raise RuntimeError("No AMD GPU found via amdsmi")
        self._amdsmi = amdsmi
        self._handle = devices[0]

    def query(self):
        a = self._amdsmi
        busy = a.amdsmi_get_gpu_activity(self._handle)
        gpu_percent = busy.get("gfx_activity", 0) or 0
        vram_info = a.amdsmi_get_gpu_vram_usage(self._handle)
        vram_used = vram_info.get("vram_used", 0) or 0
        vram_total = vram_info.get("vram_total", 0) or 0
        return gpu_percent, vram_used, vram_total


class SysfsBackend(GpuBackend):
    """Linux sysfs fallback for AMD GPUs."""
    def __init__(self):
        drm = Path("/sys/class/drm")
        for card in sorted(drm.glob("card[0-9]*")):
            vendor = card / "device" / "vendor"
            if vendor.exists() and vendor.read_text().strip() == "0x1002":
                dev = card / "device"
                if (dev / "gpu_busy_percent").exists():
                    self._device = dev
                    return
        raise RuntimeError("No AMD sysfs GPU found")

    def query(self):
        d = self._device
        gpu_pct = int((d / "gpu_busy_percent").read_text().strip())
        used = int((d / "mem_info_vram_used").read_text().strip())
        total = int((d / "mem_info_vram_total").read_text().strip())
        return gpu_pct, used, total


def init_gpu_backend():
    """Try backends in order: NVML (NVIDIA), sysfs (AMD, safe), amdsmi (AMD, native)."""
    backends = [
        ("NVML", NvmlBackend),
        ("sysfs", SysfsBackend),
        ("amdsmi", AmdsmiBackend),
    ]
    for name, cls in backends:
        try:
            instance = cls()
            instance.query()
            print(f"GPU backend: {name}")
            return instance
        except Exception:
            continue
    print("Warning: No supported GPU backend found. GPU stats will not be available.")
    return None


gpu_backend = init_gpu_backend()

class SystemStatsApp:
    def __init__(self):
        self.running = False
        self.active_generators = []
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        # Handle different shutdown signals
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, self.shutdown_handler)
        
        # Also register atexit handler as backup
        atexit.register(self.cleanup)
    
    def shutdown_handler(self, signum, frame):
        # print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        if not self.running:
            print("Cleaning up streaming connections...")
        self.running = False
        # Give a moment for generators to stop
        time.sleep(1)

    def get_system_stats(self, first = False, last_disk_io = psutil.disk_io_counters() ):

        # Set a reasonable maximum speed for the bar graph display.
        # 100 MB/s will represent a 100% full bar.
        MAX_SSD_SPEED_MB_S = 100.0
        # Get CPU and RAM stats
        if first :
            cpu_percent = psutil.cpu_percent(interval=.01)
        else:    
            cpu_percent = psutil.cpu_percent(interval=1) # This provides our 1-second delay
        memory_info = psutil.virtual_memory()
        ram_percent = memory_info.percent
        ram_used_gb = memory_info.used / (1024**3)
        ram_total_gb = memory_info.total / (1024**3)

        # Get new disk IO counters and calculate the read/write speed in MB/s
        current_disk_io = psutil.disk_io_counters()
        read_mb_s = (current_disk_io.read_bytes - last_disk_io.read_bytes) / (1024**2)
        write_mb_s = (current_disk_io.write_bytes - last_disk_io.write_bytes) / (1024**2)
        total_disk_speed = read_mb_s + write_mb_s

        # Update the last counters for the next loop
        last_disk_io = current_disk_io

        # Calculate the bar height as a percentage of our defined max speed
        ssd_bar_height = min(100.0, (total_disk_speed / MAX_SSD_SPEED_MB_S) * 100)

        # Get GPU stats
        gpu_percent, vram_percent, vram_used_gb, vram_total_gb = 0, 0, 0, 0

        if gpu_backend is not None:
            try:
                gpu_percent, vram_used, vram_total = gpu_backend.query()
                vram_used_gb = vram_used / (1024**3)
                vram_total_gb = vram_total / (1024**3)
                vram_percent = (vram_used / vram_total) * 100 if vram_total else 0
            except Exception:
                pass

        stats_html = f"""
        <style>
        .stats-container {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 0px 5px;
            height: 60px;
            width: 100%;
            box-sizing: border-box;
        }}
        
        .stats-block {{
            width: calc(18% - 5px);
            min-width: 100px;
            text-align: center;
            font-family: sans-serif;
        }}
        
        .stats-bar-background {{
            width: 90%;
            height: 30px;
            background-color: #e9ecef;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
            margin: 0 auto;
        }}
        
        .stats-bar-fill {{
            position: absolute;
            bottom: 0;
            left: 0;
            height: 100%;
            background-color: #0d6efd;
        }}
        
        .stats-title {{
            margin-top: 5px;
            font-size: 11px;
            font-weight: bold;
        }}
        
        .stats-detail {{
            font-size: 10px;
            margin-top: -2px;
        }}
        </style>
        
        <div class="stats-container">
            <!-- CPU Stat Block -->
            <div class="stats-block">
                <div class="stats-bar-background">
                    <div class="stats-bar-fill" style="width: {cpu_percent}%;"></div>
                </div>
                <div class="stats-title">CPU: {cpu_percent:.1f}%</div>
            </div>
            
            <!-- RAM Stat Block -->
            <div class="stats-block">
                <div class="stats-bar-background">
                    <div class="stats-bar-fill" style="width: {ram_percent}%;"></div>
                </div>
                <div class="stats-title">RAM {ram_percent:.1f}%</div>
                <div class="stats-detail">{ram_used_gb:.1f} / {ram_total_gb:.1f} GB</div>
            </div>
            
            <!-- SSD Activity Stat Block -->
            <div class="stats-block">
                <div class="stats-bar-background">
                    <div class="stats-bar-fill" style="width: {ssd_bar_height}%;"></div>
                </div>
                <div class="stats-title">SSD R/W</div>
                <div class="stats-detail">{read_mb_s:.1f} / {write_mb_s:.1f} MB/s</div>
            </div>
            
            <!-- GPU Stat Block -->
            <div class="stats-block">
                <div class="stats-bar-background">
                    <div class="stats-bar-fill" style="width: {gpu_percent}%;"></div>
                </div>
                <div class="stats-title">GPU: {gpu_percent:.1f}%</div>
            </div>
            
            <!-- VRAM Stat Block -->
            <div class="stats-block">
                <div class="stats-bar-background">
                    <div class="stats-bar-fill" style="width: {vram_percent}%;"></div>
                </div>
                <div class="stats-title">VRAM {vram_percent:.1f}%</div>
                <div class="stats-detail">{vram_used_gb:.1f} / {vram_total_gb:.1f} GB</div>
            </div>
        </div>
        """
        return stats_html, last_disk_io

    def streaming_html(self, state):
        if "stats_running" in state:
            return
        state["stats_running"] = True

        self.running = True
        last_disk_io = psutil.disk_io_counters()
        i = 0
        import time
        try:
            while self.running:
                i+= 1
                # if i % 2 == 0:
                #     print(f"time:{time.time()}")
                html_content, last_disk_io = self.get_system_stats(False, last_disk_io)
                yield html_content
                # time.sleep(1)
                
        except GeneratorExit:
            # print("Generator stopped gracefully")
            return
        except Exception as e:
            print(f"Streaming error: {e}")
        # finally:
        #     # Send final message indicating clean shutdown
        final_html = """
<DIV>
<img src="x" onerror="
setInterval(()=>{
    console.log('trying...');
    setTimeout(() => {
        try{
            const btn = document.getElementById('restart_stats');
            if(btn) {
                console.log('found button, clicking');
                btn.click();
            } else {
                console.log('button not found');
            }
        }catch(e){console.log('error: ' + e.message)}
    }, 100);
}, 8000);" style="display:none;">

<button onclick="document.getElementById('restart_stats').click()" 
        style="background: #007bff; color: white; padding: 15px 30px; 
               border: none; border-radius: 5px; font-size: 16px; cursor: pointer;">
   🔄 Connection to Server Lost. Attempting Auto reconnect. Click Here to for Manual Connection
</button>
</DIV>
            """
        try:
            yield final_html
        except:
            pass


    def get_gradio_element(self):
        self.system_stats_display =  gr.HTML(self.get_system_stats(True)[0])
        self.restart_btn = gr.Button("restart stats",elem_id="restart_stats", visible= False) # False)
        return self.system_stats_display
    
    def setup_events(self, main, state):
        gr.on([main.load, self.restart_btn.click],
            fn=self.streaming_html,
            inputs = state,
            outputs=self.system_stats_display,
            show_progress=False
        )
