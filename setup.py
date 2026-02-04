import os
import sys
import json
import subprocess
import argparse
import shutil
import platform

CONFIG_PATH = "setup_config.json"
IS_WIN = os.name == 'nt'

ENV_MAP = {
    "uv": {
        "dir": "env_uv", 
        "create": "uv venv --python {ver} {dir}", 
        "run": os.path.join("{dir}", "Scripts", "python.exe") if IS_WIN else os.path.join("{dir}", "bin", "python"),
        "install": (os.path.join("{dir}", "Scripts", "python.exe") if IS_WIN else os.path.join("{dir}", "bin", "python")) + " -m uv pip install"
    },
    "venv": {
        "dir": "env_venv", 
        "create": "{sys_py} -m venv {dir}", 
        "run": os.path.join("{dir}", "Scripts", "python.exe") if IS_WIN else os.path.join("{dir}", "bin", "python"),
        "install": (os.path.join("{dir}", "Scripts", "python.exe") if IS_WIN else os.path.join("{dir}", "bin", "python")) + " -m pip install"
    },
    "conda": {
        "dir": "wangp", 
        "create": "conda create -y -n {dir} python={ver}", 
        "run": "conda run -n {dir} python", 
        "install": "conda run -n {dir} pip install"
    },
    "none": {
        "dir": "", 
        "create": "", 
        "run": "python" if IS_WIN else "python3", 
        "install": "pip install"
    }
}

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: {CONFIG_PATH} not found.")
        sys.exit(1)
    with open(CONFIG_PATH, 'r') as f: return json.load(f)

def get_gpu_info():
    try:
        name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], encoding='utf-8').strip()
        return name, "NVIDIA"
    except: pass

    if IS_WIN:
        try:
            name = subprocess.check_output("wmic path win32_VideoController get name", shell=True, encoding='utf-8')
            name = name.replace("Name", "").strip().split('\n')[0].strip()
            if "Radeon" in name or "AMD" in name: return name, "AMD"
            return name, "INTEL"
        except: pass
    else:
        try:
            name = subprocess.check_output("lspci | grep -i vga", shell=True, encoding='utf-8')
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

def list_installs():
    found = []
    for k, v in ENV_MAP.items():
        if k in ["conda", "none"]: continue
        py_path = v["run"].format(dir=v["dir"])
        if os.path.exists(py_path): found.append(k)
    try:
        res = subprocess.check_output("conda env list", shell=True, encoding='utf-8')
        if "wangp " in res: found.append("conda")
    except: pass
    return found

def get_active_env():
    installs = list_installs()
    if not installs: return "none"
    if len(installs) == 1: return installs[0]
    print("\nMultiple environments detected:")
    for i, env in enumerate(installs): print(f"{i+1}. {env}")
    choice = input("Select environment to use: ")
    return installs[int(choice)-1]

def install_from_keys(env_type, py_k, torch_k, triton_k, sage_k, flash_k, kernel_list, config):
    env_dir = ENV_MAP[env_type]["dir"]
    target_py_ver = config['components']['python'][py_k]['ver']
    
    print(f"\n[1/3] Preparing Environment: {env_type} (Python {target_py_ver})...")
    if env_type != "none":
        if env_type == "conda": run_cmd(f"conda env remove -y -n {env_dir}")
        elif os.path.exists(env_dir): shutil.rmtree(env_dir)
        
        create_cmd = ENV_MAP[env_type]["create"].format(ver=target_py_ver, dir=env_dir, sys_py=sys.executable)
        run_cmd(create_cmd)

    pip = ENV_MAP[env_type]["install"].format(dir=env_dir)
    
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
                act = f". {env_dir}/bin/activate && " if not IS_WIN else ""
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

def do_migrate(config):
    print("\n" + "="*60)
    print("      WAN2GP AUTOMATED PLATFORM MIGRATION (TO 3.11)")
    print("="*60)
    env = get_active_env()
    if env == "none":
        print("No environment found. Please run install first.")
        return

    confirm = input(f"This will wipe your {env} and rebuild. Proceed? (y/n): ")
    if confirm.lower() != 'y': return

    target = config['gpu_profiles']['RTX_50'] 
    install_from_keys(env, target['python'], target['torch'], target['triton'], target['sage'], target.get('flash'), target['kernels'], config)

def do_upgrade(config):
    print("\n" + "="*60)
    print("      WAN2GP MANUAL COMPONENT UPGRADE")
    print("="*60)
    env = get_active_env()
    gpu_name, vendor = get_gpu_info()
    rec = config['gpu_profiles'][get_profile_key(gpu_name, vendor)]

    py_k = menu("Python Version", config['components']['python'], rec['python'])
    torch_k = menu("Torch Version", config['components']['torch'], rec['torch'])
    triton_k = menu("Triton", config['components']['triton'], rec['triton'])
    sage_k = menu("Sage Attention", config['components']['sage'], rec['sage'])
    flash_k = menu("Flash Attention", config['components']['flash'], rec['flash'])
    
    install_from_keys(env, py_k, torch_k, triton_k, sage_k, flash_k, rec['kernels'], config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["install", "run", "update", "migrate", "upgrade"])
    parser.add_argument("--env", default="venv")
    args = parser.parse_args()
    cfg = load_config()
    
    gpu_name, vendor = get_gpu_info()
    profile_key = get_profile_key(gpu_name, vendor)
    profile = cfg['gpu_profiles'][profile_key]

    if args.mode == "install":
        print(f"Hardware Detected: {gpu_name} ({vendor})")
        install_from_keys(args.env, profile['python'], profile['torch'], profile['triton'], profile['sage'], profile.get('flash'), profile['kernels'], cfg)
    
    elif args.mode == "run":
        env = get_active_env()
        env_vars = profile.get("env", {})
        cmd_fmt = ENV_MAP[env]['run']
        cmd = f"{cmd_fmt.format(dir=ENV_MAP[env]['dir'])} wgp.py"
        run_cmd(cmd, env_vars=env_vars)

    elif args.mode == "update":
        run_cmd("git pull")
        env = get_active_env()
        cmd_fmt = ENV_MAP[env]['run']
        cmd = f"{cmd_fmt.format(dir=ENV_MAP[env]['dir'])} -m pip install -r requirements.txt"
        run_cmd(cmd)

    elif args.mode == "migrate":
        do_migrate(cfg)

    elif args.mode == "upgrade":
        do_upgrade(cfg)