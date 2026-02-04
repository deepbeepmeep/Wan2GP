import os
import sys
import json
import subprocess
import argparse
import shutil

CONFIG_PATH = "setup_config.json"
ENV_MAP = {
    "uv": {"dir": "env_uv", "create": "uv venv --python {ver} {dir}", "run": r"{dir}\Scripts\python.exe", "install": r"{dir}\Scripts\python.exe -m uv pip install"},
    "venv": {"dir": "env_venv", "create": "{sys_py} -m venv {dir}", "run": r"{dir}\Scripts\python.exe", "install": r"{dir}\Scripts\python.exe -m pip install"},
    "conda": {"dir": "wangp", "create": "conda create -y -n {dir} python={ver}", "run": "conda run -n {dir} python", "install": "conda run -n {dir} pip install"},
    "none": {"dir": "", "create": "", "run": "python", "install": "pip install"}
}

def load_config():
    with open(CONFIG_PATH, 'r') as f: return json.load(f)

def get_gpu_name():
    try: return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], encoding='utf-8').strip()
    except: return None

def get_profile_key(gpu):
    if not gpu: return "RTX_40"
    g = gpu.upper()
    if "50" in g: return "RTX_50"
    if "40" in g: return "RTX_40"
    if "30" in g: return "RTX_30"
    if "20" in g or "QUADRO" in g: return "RTX_20"
    return "GTX_10"

def run_cmd(cmd):
    if not cmd: return
    print(f"\n>>> Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def list_installs():
    found = []
    for k, v in ENV_MAP.items():
        if k in ["conda", "none"]: continue
        if os.path.exists(os.path.join(v["dir"], "Scripts", "python.exe")): found.append(k)
    try:
        res = subprocess.check_output("conda env list", shell=True, encoding='utf-8')
        if "wangp " in res: found.append("conda")
    except: pass
    return found

def get_active_env():
    installs = list_installs()
    return installs[0] if installs else "none"

def install_from_keys(env_type, py_k, torch_k, triton_k, sage_k, flash_k, kernel_list, config):
    env_dir = ENV_MAP[env_type]["dir"]
    target_py_ver = config['components']['python'][py_k]['ver']
    
    print(f"\n[1/3] Preparing Environment: {env_type} (Python {target_py_ver})...")
    if env_type != "none":
        if env_type == "conda": run_cmd(f"conda env remove -y -n {env_dir}")
        elif os.path.exists(env_dir): shutil.rmtree(env_dir)
        run_cmd(ENV_MAP[env_type]["create"].format(ver=target_py_ver, dir=env_dir, sys_py=sys.executable))

    pip = ENV_MAP[env_type]["install"].format(dir=env_dir)
    
    print(f"\n[2/3] Installing Torch: {config['components']['torch'][torch_k]['label']}...")
    run_cmd(f"{pip} {config['components']['torch'][torch_k]['cmd']}")
    
    print(f"\n[3/3] Installing Requirements & Extras...")
    run_cmd(f"{pip} -r requirements.txt")
    if triton_k: run_cmd(f"{pip} {config['components']['triton'][triton_k]['cmd']}")
    if sage_k: run_cmd(f"{pip} {config['components']['sage'][sage_k]['cmd']}")
    if flash_k: run_cmd(f"{pip} {config['components']['flash'][flash_k]['cmd']}")
    for k in kernel_list: run_cmd(f"{pip} {config['components']['kernels'][k]['cmd']}")

def menu(title, options, recommended_key=None):
    print(f"\n--- {title} ---")
    keys = list(options.keys())
    for i, k in enumerate(keys):
        rec = " [RECOMMENDED]" if k == recommended_key else ""
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
        print("No environment found. Please run install.bat first.")
        return
    
    confirm = input(f"This will nuke your {env} and rebuild with Python 3.11 / Torch 2.10.\nProceed? (y/n): ")
    if confirm.lower() != 'y': return
    
    target = config['gpu_profiles']['RTX_50']
    install_from_keys(env, target['python'], target['torch'], target['triton'], target['sage'], target['flash'], target['kernels'], config)

def do_upgrade(config):
    print("\n" + "="*60)
    print("      WAN2GP MANUAL COMPONENT UPGRADE")
    print("="*60)
    env = get_active_env()
    gpu = get_gpu_name()
    rec = config['gpu_profiles'][get_profile_key(gpu)]

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
    
    if args.mode == "install":
        p = cfg['gpu_profiles'][get_profile_key(get_gpu_name())]
        install_from_keys(args.env, p['python'], p['torch'], p['triton'], p['sage'], p['flash'], p['kernels'], cfg)
    elif args.mode == "run":
        env = get_active_env()
        run_cmd(f"{ENV_MAP[env]['run'].format(dir=ENV_MAP[env]['dir'])} wgp.py")
    elif args.mode == "update":
        run_cmd("git pull")
        env = get_active_env()
        run_cmd(f"{ENV_MAP[env]['run'].format(dir=ENV_MAP[env]['dir'])} -m pip install -r requirements.txt")
    elif args.mode == "migrate":
        do_migrate(cfg)
    elif args.mode == "upgrade":
        do_upgrade(cfg)