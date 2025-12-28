import time
import threading
import torch

gen_lock = threading.Lock()

def get_gen_info(state):
    cache = state.get("gen", None)
    if cache == None:
        cache = dict()
        state["gen"] = cache
    return cache

def any_GPU_process_running(state, process_id, ignore_main = False):
    gen = get_gen_info(state)
#"process:" + process_id
    with gen_lock:
        process_status = gen.get("process_status", None)
        return process_status is not None and not (process_status =="process:main" and ignore_main)
    
def acquire_GPU_ressources(state, process_id, process_name, gr = None, custom_pause_msg = None, custom_wait_msg = None):
    gen = get_gen_info(state)
    original_process_status = None
    while True:
        with gen_lock:
            process_hierarchy = gen.get("process_hierarchy", None)
            if process_hierarchy is None:
                process_hierarchy = dict()
                gen["process_hierarchy"]= process_hierarchy

            process_status = gen.get("process_status", None)
            if process_status is None:
                original_process_status = process_status 
                gen["process_status"] = "process:" + process_id
                break
            elif process_status == "process:main":
                original_process_status = process_status 
                gen["process_status"] = "request:" + process_id

                gen["pause_msg"] = custom_pause_msg if custom_pause_msg is not None else f"Generation Suspended while using {process_name}" 
                break
            elif process_status == "process:" + process_id:
                break
        time.sleep(0.1)

    if original_process_status is not None:
        total_wait = 0
        wait_time = 0.1
        wait_msg_displayed = False
        while True:
            with gen_lock:
                process_status = gen.get("process_status", None)
                if process_status == "process:" + process_id: break
                if process_status is None:
                    # handle case when main process has finished at some point in between the last check and now
                    gen["process_status"] = "process:" + process_id
                    break

            total_wait += wait_time
            if round(total_wait,2) >= 5 and gr is not None and not wait_msg_displayed:
                wait_msg_displayed = True
                if custom_wait_msg is None:
                    gr.Info(f"Process {process_name} is Suspended while waiting that GPU Ressources become available")
                else:
                    gr.Info(custom_wait_msg)

            time.sleep(wait_time)
    
    with gen_lock:
        process_hierarchy[process_id] = original_process_status
    torch.cuda.synchronize()

def release_GPU_ressources(state, process_id):
    gen = get_gen_info(state)
    torch.cuda.synchronize()
    with gen_lock:
        process_hierarchy = gen.get("process_hierarchy", {})
        gen["process_status"] = process_hierarchy.get(process_id, None)
