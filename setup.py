import os
import sys
import json
import subprocess
import argparse
import shutil
import platform

CONFIG_PATH = "setup_config.json"
ENVS_FILE = "envs.json"
IS_WIN = os.name == 'nt'

ENV_TEMPLATES = {
    "uv": {
        "create": "uv venv --python {ver} \"{dir}\"",
        "run": os.path.join("{dir}", "Scripts", "python.exe") if IS_WIN else os.path.join("{dir}", "bin", "python"),
        "install": (os.path.join("{dir}", "Scripts", "python.exe") if IS_WIN else os.path.join("{dir}", "bin", "python")) + " -m uv pip install"
    },
    "venv": {
        "create": "{sys_py} -m venv \"{dir}\"",
        "run": os.path.join("{dir}", "Scripts", "python.exe") if IS_WIN else os.path.join("{dir}", "bin", "python"),
        "install": (os.path.join("{dir}", "Scripts", "python.exe") if IS_WIN else os.path.join("{dir}", "bin", "python")) + " -m pip install"
    },
    "conda": {
        "create": "conda create -y -p \"{dir}\" python={ver}",
        "run": "conda run -p \"{dir}\" python",
        "install": "conda run -p \"{dir}\" pip install"
    },
    "none": {
        "create": "",
        "run": "python" if IS_WIN else "python3",
        "install": "pip install"
    }
}

VERSION_CHECK_SCRIPT = """
import sys
import importlib
import importlib.metadata

pkgs = ['torch', 'triton', 'sageattention', 'flash_attn']
res = []
try:
    res.append(f"python={sys.version.split()[0]}")
except: 
    res.append("python=Unknown")

for p in pkgs:
    try:
        ver = importlib.metadata.version(p)
        res.append(f"{p}={ver}")
    except importlib.metadata.PackageNotFoundError:
        try:
            # Fallback to __version__
            m = importlib.import_module(p)
            ver = getattr(m, '__version__', 'Installed')
            res.append(f"{p}={ver}")
        except ImportError:
            res.append(f"{p}=Missing")
    except Exception:
        res.append(f"{p}=Error")
print("||".join(res))
"""

class EnvsManager:
    def __init__(self):
        self.data = {"active": None, "envs": {}}
        self.load()

    def load(self):
        if os.path.exists(ENVS_FILE):
            try:
                with open(ENVS_FILE, 'r') as f:
                    self.data = json.load(f)
            except:
                print(f"[!] Warning: {ENVS_FILE} corrupted. Starting fresh.")

    def save(self):
        with open(ENVS_FILE, 'w') as f:
            json.dump(self.data, f, indent=4)

    def get_active(self):
        return self.data.get("active")

    def set_active(self, name):
        if name in self.data["envs"]:
            self.data["active"] = name
            self.save()
            print(f"[*] '{name}' is now the active environment.")
        else:
            print(f"[!] Environment '{name}' not found.")

    def add_env(self, name, type, path):
        self.data["envs"][name] = {"type": type, "path": path}
        if not self.data["active"]:
            self.data["active"] = name
        self.save()

    def remove_env(self, name):
        if name in self.data["envs"]:
            entry = self.data["envs"][name]
            path = entry["path"]

            if os.path.exists(path) and entry["type"] != "none":
                try:
                    print(f"[*] Deleting directory: {path}")
                    if entry["type"] == "conda":
                         run_cmd(f"conda env remove -p \"{path}\" -y")
                    else:
                        shutil.rmtree(path)
                except Exception as e:
                    print(f"[!] Error removing directory: {e}")

            del self.data["envs"][name]
            if self.data["active"] == name:
                self.data["active"] = None
                keys = list(self.data["envs"].keys())
                if keys:
                    self.data["active"] = keys[0]
                    print(f"[*] Active environment switched to '{keys[0]}'.")
                else:
                    print("[*] No environments left.")
            self.save()

    def list_envs(self):
        return self.data["envs"]

    def resolve_target_env(self):
        """Intelligently determine which env to use for operations."""
        envs = self.list_envs()
        if not envs:
            print("[!] No environments found. Please run install first.")
            sys.exit(1)
        
        active = self.get_active()

        if len(envs) == 1:
            return list(envs.keys())[0]

        print("\nMultiple environments detected:")
        keys = list(envs.keys())
        for i, k in enumerate(keys):
            marker = "*" if k == active else " "
            print(f"{i+1}. [{marker}] {k} ({envs[k]['type']})")
        
        print(f"Default: {active}")
        choice = input("Select environment (Number) or Press Enter for Default: ").strip()
        
        if choice == "":
            return active
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        except:
            pass
        return active

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: {CONFIG_PATH} not found.")
        sys.exit(1)
    with open(CONFIG_PATH, 'r') as f: return json.load(f)

def get_gpu_info():
    try:
        name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
            encoding='utf-8', 
            stderr=subprocess.DEVNULL
        ).strip()
        return name, "NVIDIA"
    except: pass

    if IS_WIN:
        try:
            name = subprocess.check_output(
                "wmic path win32_VideoController get name", 
                shell=True, 
                encoding='utf-8', 
                stderr=subprocess.DEVNULL
            )
            name = name.replace("Name", "").strip().split('\n')[0].strip()
            if "Radeon" in name or "AMD" in name: return name, "AMD"
            return name, "INTEL"
        except: pass
    else:
        try:
            name = subprocess.check_output(
                "lspci | grep -i vga", 
                shell=True, 
                encoding='utf-8', 
                stderr=subprocess.DEVNULL
            )
            if "NVIDIA" in name: return name, "NVIDIA"
            if "AMD" in name or "Advanced Micro Devices" in name: return name, "AMD"
        except: pass

    return "Unknown", "UNKNOWN"

def get_profile_key(gpu_name, vendor):
    g = gpu_name.upper()
    if vendor == "NVIDIA":
        if "50" in g: return "RTX_50"
        if "40" in g: return "RTX_40"
        if "30" in g: return "RTX_30"
        if "20" in g or "QUADRO" in g: return "RTX_20"
        return "GTX_10"
    elif vendor == "AMD":
        if any(x in g for x in ["7600", "7700", "7800", "7900"]): return "AMD_GFX110X"
        if any(x in g for x in ["7000", "Z1", "PHOENIX"]): return "AMD_GFX1151"
        if any(x in g for x in ["8000", "STRIX", "1201"]): return "AMD_GFX1201"
        return "AMD_GFX110X" 
    return "RTX_40"

def get_os_key():
    return "win" if IS_WIN else "linux"

def resolve_cmd(cmd_entry):
    if isinstance(cmd_entry, dict):
        return cmd_entry.get(get_os_key())
    return cmd_entry

def run_cmd(cmd, env_vars=None):
    if not cmd: return

    if "&&" in cmd and not IS_WIN:
        print(f"\n>>> Running (Shell): {cmd}")
        custom_env = os.environ.copy()
        if env_vars: custom_env.update(env_vars)
        subprocess.run(cmd, shell=True, check=True, env=custom_env)
        return

    print(f"\n>>> Running: {cmd}")
    custom_env = os.environ.copy()
    if env_vars:
        for k, v in env_vars.items():
            print(f"    [ENV SET] {k}={v}")
            custom_env[k] = v

    subprocess.run(cmd, shell=True, check=True, env=custom_env)

def get_env_details(name, env_data):
    env_type = env_data["type"]
    dir_name = env_data["path"]
    entry = ENV_TEMPLATES[env_type]
    
    if env_type == "conda":
        cmd_base = entry['run'].format(dir=dir_name)
        full_cmd = f"{cmd_base} -c \"{VERSION_CHECK_SCRIPT.replace(chr(10), ';')}\""
    else:
        py_exec = entry['run'].format(dir=dir_name)
        full_cmd = [py_exec, "-c", VERSION_CHECK_SCRIPT]
        
    try:
        if env_type == "conda":
            output = subprocess.check_output(full_cmd, shell=True, encoding='utf-8', stderr=subprocess.DEVNULL)
        else:
            output = subprocess.check_output(full_cmd, encoding='utf-8', stderr=subprocess.DEVNULL)
        
        data = {k: v for k, v in [x.split('=') for x in output.strip().split('||')]}
        data['path'] = dir_name
        data['type'] = env_type
        return data
    except Exception as e:
        return {'error': str(e), 'type': env_type, 'path': dir_name}

def show_status():
    manager = EnvsManager()
    print("\n" + "="*95)
    print(f"{'INSTALLED ENVIRONMENTS & VERSIONS':^95}")
    print("="*95)
    
    envs = manager.list_envs()
    active = manager.get_active()
    
    if not envs:
        print("   No environments installed.")
        print("="*95)
        return

    print(f"{'NAME':<15} | {'TYPE':<6} | {'PYTHON':<8} | {'TORCH':<15} | {'TRITON':<10} | {'SAGE':<12} | {'FLASH':<12}")
    print("-" * 95)

    for name, data in envs.items():
        details = get_env_details(name, data)
        marker = "*" if name == active else " "
        display_name = f"[{marker}] {name}"
        
        if 'error' in details:
            print(f"{display_name:<15} | {data['type']:<6} | [Error reading environment]")
            continue
            
        print(f"{display_name:<15} | {data['type']:<6} | "
              f"{details.get('python','?'):<8} | "
              f"{details.get('torch','?'):<15} | "
              f"{details.get('triton','?'):<10} | "
              f"{details.get('sageattention','?'):<12} | "
              f"{details.get('flash_attn','?'):<12}")
    
    print("-" * 95)
    print(f" * = Active Environment")
    print("="*95 + "\n")

def install_logic(env_name, env_type, env_path, py_k, torch_k, triton_k, sage_k, flash_k, kernel_list, config):
    template = ENV_TEMPLATES[env_type]
    target_py_ver = config['components']['python'][py_k]['ver']
    
    print(f"\n[1/3] Preparing Environment: {env_name} ({env_type})...")

    if env_type != "none":
        create_cmd = template["create"].format(ver=target_py_ver, dir=env_path, sys_py=sys.executable)
        if create_cmd:
            run_cmd(create_cmd)

    pip = template["install"].format(dir=env_path)
    
    print(f"\n[2/3] Installing Torch: {config['components']['torch'][torch_k]['label']}...")
    torch_cmd = resolve_cmd(config['components']['torch'][torch_k]['cmd'])
    run_cmd(f"{pip} {torch_cmd}")
    
    print(f"\n[3/3] Installing Requirements & Extras...")
    run_cmd(f"{pip} -r requirements.txt")
    
    if triton_k: 
        cmd = resolve_cmd(config['components']['triton'][triton_k]['cmd'])
        if cmd: run_cmd(f"{pip} {cmd}")
        
    if sage_k: 
        cmd = resolve_cmd(config['components']['sage'][sage_k]['cmd'])
        if cmd.startswith("http") or cmd.startswith("sageattention"):
            run_cmd(f"{pip} {cmd}")
        else:
            if env_type == "venv" or env_type == "uv":
                act = f". {env_path}/bin/activate && " if not IS_WIN else ""
                run_cmd(f"{act}{cmd}")
            elif env_type == "conda":
                pass

    if flash_k: 
        cmd = resolve_cmd(config['components']['flash'][flash_k]['cmd'])
        if cmd: run_cmd(f"{pip} {cmd}")
        
    for k in kernel_list:
        if k in config['components']['kernels']:
            cmd = resolve_cmd(config['components']['kernels'][k]['cmd'])
            if cmd: run_cmd(f"{pip} {cmd}")

def menu(title, options, recommended_key=None):
    print(f"\n--- {title} ---")
    keys = list(options.keys())
    for i, k in enumerate(keys):
        rec = " [RECOMMENDED FOR YOUR GPU]" if k == recommended_key else ""
        print(f"{i+1}. {options[k]['label']}{rec}")
    choice = input(f"Select option (Enter for Recommended): ")
    if choice == "" and recommended_key: return recommended_key
    try: return keys[int(choice)-1]
    except: return recommended_key

def do_install_interactive(env_type, config, detected_key):
    manager = EnvsManager()
    create_wgp_config(detected_key, config)

    default_name = f"env_{env_type}" if env_type != "none" else "system"
    print(f"\n--- Configuration for {env_type} ---")
    name = input(f"Enter a name for this environment (Default: {default_name}): ").strip()
    if not name: name = default_name
    
    cwd = os.getcwd()
    path = os.path.join(cwd, name) if env_type != "none" else ""
    
    if name in manager.list_envs():
        print(f"\n[!] Warning: Environment '{name}' already exists in registry.")
        choice = input("Do you want to overwrite it? (This will delete the old folder) [y/N]: ").lower()
        if choice != 'y': return
        manager.remove_env(name)
    elif os.path.exists(path) and env_type != "none":
        print(f"\n[!] Warning: Directory '{path}' exists but is not registered.")
        choice = input("Do you want to overwrite this directory? [y/N]: ").lower()
        if choice != 'y': return
        try: shutil.rmtree(path)
        except: pass

    print("\n--- Select Install Mode ---")
    print("1. Autoselect (Recommended - Based on your card)")
    print("2. Manual Selection (Custom versions)")
    print("3. Use Latest (Forces RTX 50 Profile)")
    
    mode = input("Select option (1-3) [Default: 1]: ").strip()
    
    if mode == "2":
        base = config['gpu_profiles'][detected_key]
        py_k = menu("Python Version", config['components']['python'], base['python'])
        torch_k = menu("Torch Version", config['components']['torch'], base['torch'])
        triton_k = menu("Triton", config['components']['triton'], base['triton'])
        sage_k = menu("Sage Attention", config['components']['sage'], base['sage'])
        flash_k = menu("Flash Attention", config['components']['flash'], base['flash'])
        kernels = base['kernels']
        
        install_logic(name, env_type, path, py_k, torch_k, triton_k, sage_k, flash_k, kernels, config)
        
    elif mode == "3":
        p = config['gpu_profiles']['RTX_50']
        install_logic(name, env_type, path, p['python'], p['torch'], p['triton'], p['sage'], p.get('flash'), p['kernels'], config)
    else:
        p = config['gpu_profiles'][detected_key]
        install_logic(name, env_type, path, p['python'], p['torch'], p['triton'], p['sage'], p.get('flash'), p['kernels'], config)

    manager.add_env(name, env_type, path)
    
    if len(manager.list_envs()) > 1:
        choice = input(f"\nDo you want to make '{name}' the active environment? [Y/n]: ").lower()
        if choice != 'n':
            manager.set_active(name)
    else:
        print(f"\n[*] '{name}' is the only environment, setting as active.")
        manager.set_active(name)

def do_manage():
    manager = EnvsManager()
    while True:
        os.system('cls' if IS_WIN else 'clear')
        print("======================================================")
        print("               ENVIRONMENT MANAGER")
        print("======================================================")
        envs = manager.list_envs()
        active = manager.get_active()
        
        if not envs:
            print(" No environments installed.")
        else:
            for name, data in envs.items():
                status = "(Active)" if name == active else ""
                print(f" - {name:<15} [{data['type']}] {status}")
        
        print("------------------------------------------------------")
        print("1. Set Active Environment")
        print("2. Delete Environment")
        print("3. Add Existing Environment")
        print("4. List Environment Details")
        print("5. Return to Menu / Exit")
        
        choice = input("\nSelect option: ")
        
        if choice == "1":
            name = input("Enter name of environment to activate: ")
            manager.set_active(name)
            input("Press Enter...")
        elif choice == "2":
            name = input("Enter name of environment to DELETE: ")
            conf = input(f"Are you sure you want to delete '{name}' and its files? (y/n): ")
            if conf.lower() == 'y':
                manager.remove_env(name)
                input("Deleted. Press Enter...")
        elif choice == "3":
            path = input("Enter the path to the existing environment folder: ").strip()
            if not os.path.exists(path):
                print("[!] Error: Path does not exist.")
            else:
                name = input("Enter a nickname for this environment: ").strip()
                if not name: name = os.path.basename(path.rstrip(os.sep))
                
                print("\nSelect Environment Type:")
                print("1. venv")
                print("2. uv")
                print("3. conda")
                t_choice = input("Choice (Default 1): ")
                e_type = "uv" if t_choice == "2" else "conda" if t_choice == "3" else "venv"
                
                manager.add_env(name, e_type, os.path.abspath(path))
                print(f"[*] Registered '{name}' at {os.path.abspath(path)}")
            input("Press Enter...")
        elif choice == "4":
            show_status()
            input("Press Enter...")
        elif choice == "5":
            break

def do_migrate(config):
    manager = EnvsManager()
    print("\n" + "="*60)
    print("      WAN2GP AUTOMATED PLATFORM MIGRATION (TO 3.11)")
    print("="*60)
    
    env_name = manager.resolve_target_env()
    env_data = manager.list_envs()[env_name]
    
    print(f"\nTarget Environment: {env_name} ({env_data['type']})")
    confirm = input(f"This will wipe '{env_name}' and rebuild it. Proceed? (y/n): ")
    if confirm.lower() != 'y': return

    target = config['gpu_profiles']['RTX_50']

    manager.remove_env(env_name)

    install_logic(env_name, env_data['type'], env_data['path'], 
                  target['python'], target['torch'], target['triton'], 
                  target['sage'], target.get('flash'), target['kernels'], config)
    
    manager.add_env(env_name, env_data['type'], env_data['path'])

def do_upgrade(config):
    manager = EnvsManager()
    print("\n" + "="*60)
    print("      WAN2GP MANUAL COMPONENT UPGRADE")
    print("="*60)
    
    env_name = manager.resolve_target_env()
    env_data = manager.list_envs()[env_name]
    
    gpu_name, vendor = get_gpu_info()
    rec = config['gpu_profiles'][get_profile_key(gpu_name, vendor)]

    py_k = menu("Python Version", config['components']['python'], rec['python'])
    torch_k = menu("Torch Version", config['components']['torch'], rec['torch'])
    triton_k = menu("Triton", config['components']['triton'], rec['triton'])
    sage_k = menu("Sage Attention", config['components']['sage'], rec['sage'])
    flash_k = menu("Flash Attention", config['components']['flash'], rec['flash'])

    install_logic(env_name, env_data['type'], env_data['path'], py_k, torch_k, triton_k, sage_k, flash_k, rec['kernels'], config)

def get_system_specs():
    ram_gb = 0
    vram_gb = 0

    if IS_WIN:
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                encoding='utf-8', stderr=subprocess.DEVNULL
            ).strip()
            if out:
                ram_gb = int(out) / (1024**3)
        except:
            try:
                out = subprocess.check_output(
                    "wmic computersystem get TotalPhysicalMemory /value", 
                    shell=True, encoding='utf-8', stderr=subprocess.DEVNULL
                )
                for line in out.splitlines():
                    if "TotalPhysicalMemory=" in line:
                        ram_gb = int(line.split('=')[1]) / (1024**3)
                        break
            except:
                pass
    else:
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        kb_val = float(line.split()[1])
                        ram_gb = kb_val / (1024**2)
                        break
        except: pass

    if ram_gb == 0:
        print("[!] Warning: Could not detect System RAM. Defaulting to 16GB.")
        ram_gb = 16

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], 
            encoding='utf-8', stderr=subprocess.DEVNULL
        ).strip()
        vram_gb = float(out.split('\n')[0]) / 1024
    except:
        print("[!] Warning: Could not detect VRAM via nvidia-smi. Defaulting to 8GB.")
        vram_gb = 8
        
    return ram_gb, vram_gb

def create_wgp_config(profile_key, config_data):
    WGP_CONFIG_FILE = "wgp_config.json"
    
    if os.path.exists(WGP_CONFIG_FILE):
        return

    print("\n[*] Auto-generating wgp_config.json based on hardware...")
    
    ram, vram = get_system_specs()
    print(f"    Detected: {int(ram)}GB RAM / {int(vram)}GB VRAM")

    has_high_ram = ram > 60
    has_mid_ram = ram > 30
    has_huge_vram = vram > 22
    has_high_vram = vram > 11
    
    pid = 5
    
    if has_high_ram and has_huge_vram:
        pid = 1
    elif has_high_ram: 
        pid = 2
    elif has_mid_ram and has_huge_vram:
        pid = 3 
    elif has_mid_ram and has_high_vram:
        pid = 4
    else:
        pid = 5

    prof_settings = config_data['gpu_profiles'].get(profile_key, {})

    attn_mode = ""
    if "50" in profile_key or "40" in profile_key or "30" in profile_key:
        attn_mode = "sage2"
    elif "20" in profile_key:
        attn_mode = "sage"

    compile_mode = ""
    triton_key = prof_settings.get('triton')
    if triton_key and triton_key != "none":
        compile_mode = "transformer"

    config_out = {
        "attention_mode": attn_mode,
        "compile": compile_mode,
        "video_profile": pid,
        "image_profile": pid,
        "audio_profile": pid,
    }
    
    try:
        with open(WGP_CONFIG_FILE, 'w') as f:
            json.dump(config_out, f, indent=4)
        print(f"    Created config with Profile {pid}, Attention: '{attn_mode}', Compile: '{compile_mode}'")
    except Exception as e:
        print(f"[!] Error writing config: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["install", "run", "update", "migrate", "upgrade", "status", "manage"])
    parser.add_argument("--env", default="venv", help="Type of env for install (venv, uv, conda, none)")
    args = parser.parse_args()
    cfg = load_config()
    
    if args.mode == "status":
        show_status()
        sys.exit(0)

    if args.mode == "manage":
        do_manage()
        sys.exit(0)

    gpu_name, vendor = get_gpu_info()
    profile_key = get_profile_key(gpu_name, vendor)
    profile = cfg['gpu_profiles'][profile_key]

    if args.mode == "install":
        print(f"Hardware Detected: {gpu_name} ({vendor})")
        do_install_interactive(args.env, cfg, profile_key)
    
    elif args.mode == "run":
        manager = EnvsManager()
        active = manager.get_active()
        if not active:
            print("[!] No active environment found. Run install or manage.")
            sys.exit(1)
        
        env_data = manager.list_envs().get(active)
        if not env_data:
            print(f"[!] Active environment '{active}' data missing from registry.")
            sys.exit(1)

        print(f"[*] Launching using active environment: {active}")

        extra_args = ""
        if os.path.exists("scripts/args.txt"):
            with open("scripts/args.txt", "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
                extra_args = " ".join(lines)
        
        env_vars = profile.get("env", {})
        cmd_fmt = ENV_TEMPLATES[env_data['type']]['run']
        cmd = f"{cmd_fmt.format(dir=env_data['path'])} wgp.py {extra_args}"
        run_cmd(cmd, env_vars=env_vars)

    elif args.mode == "update":
        run_cmd("git pull")
        manager = EnvsManager()
        env_name = manager.resolve_target_env()
        env_data = manager.list_envs()[env_name]
        
        cmd_fmt = ENV_TEMPLATES[env_data['type']]['run']
        cmd = f"{cmd_fmt.format(dir=env_data['path'])} -m pip install -r requirements.txt"
        run_cmd(cmd)

    elif args.mode == "migrate":
        do_migrate(cfg)

    elif args.mode == "upgrade":
        do_upgrade(cfg)