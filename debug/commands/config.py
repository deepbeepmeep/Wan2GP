"""Config display command."""

import os
from dotenv import load_dotenv


def _format_secret_status(value: str | None) -> str:
    return "set" if value else "Not set"


def run(client, options: dict):
    """Handle 'debug.py config' command."""
    load_dotenv()
    
    explain = options.get('explain', False)
    
    print("=" * 80)
    print("⚙️  SYSTEM CONFIGURATION")
    print("=" * 80)
    
    # Timing configurations
    configs = [
        ("ORCHESTRATOR_POLL_SEC", "30", "How often orchestrator checks workers"),
        ("WORKER_GRACE_PERIOD_SEC", "120", "Grace period before idle check"),
        ("GPU_IDLE_TIMEOUT_SEC", "300", "Heartbeat timeout for active workers"),
        ("GPU_OVERCAPACITY_IDLE_TIMEOUT_SEC", "30", "Idle timeout when over-capacity"),
        ("MIN_ACTIVE_GPUS", "2", "Minimum workers to keep running"),
        ("MAX_ACTIVE_GPUS", "10", "Maximum workers allowed"),
        ("TASKS_PER_GPU_THRESHOLD", "3", "Tasks per GPU before scaling up"),
    ]
    
    print("\n📊 Timing & Scaling Configuration")
    print("-" * 80)
    
    for env_var, default, description in configs:
        value = os.getenv(env_var, default)
        print(f"\n{env_var}: {value}")
        if explain:
            print(f"  └─ {description}")
            if env_var == "WORKER_GRACE_PERIOD_SEC":
                mins = int(value) / 60
                print(f"  └─ {mins:.1f} minutes after promotion to active")
            elif env_var == "ORCHESTRATOR_POLL_SEC":
                print(f"  └─ Workers checked every {value} seconds")
    
    # RunPod configuration
    print("\n\n☁️  RunPod Configuration")
    print("-" * 80)
    
    runpod_configs = [
        ("RUNPOD_API_KEY", "API key for RunPod"),
        ("RUNPOD_TEMPLATE_ID", "Default template for GPU workers"),
        ("RUNPOD_NETWORK_VOLUME_ID", "Network volume for shared storage"),
    ]
    
    for env_var, description in runpod_configs:
        value = os.getenv(env_var, "Not set")
        if value and value != "Not set" and "KEY" in env_var:
            display_value = _format_secret_status(value)
        else:
            display_value = value
        print(f"\n{env_var}: {display_value}")
        if explain:
            print(f"  └─ {description}")
    
    # Database configuration
    print("\n\n🗄️  Database Configuration")
    print("-" * 80)
    
    db_configs = [
        ("SUPABASE_URL", "Supabase project URL"),
        ("SUPABASE_SERVICE_ROLE_KEY", "Service role key (admin access)"),
    ]
    
    for env_var, description in db_configs:
        value = os.getenv(env_var, "Not set")
        if value and value != "Not set" and "KEY" in env_var:
            display_value = _format_secret_status(value)
        else:
            display_value = value
        print(f"\n{env_var}: {display_value}")
        if explain:
            print(f"  └─ {description}")
    
    # Termination logic explanation
    if explain:
        print("\n\n📋 Worker Termination Logic")
        print("-" * 80)
        grace = int(os.getenv("WORKER_GRACE_PERIOD_SEC", "120"))
        poll = int(os.getenv("ORCHESTRATOR_POLL_SEC", "30"))
        min_gpus = os.getenv("MIN_ACTIVE_GPUS", "2")
        
        print(f"""
1. Worker is promoted to 'active' status
2. Grace period starts ({grace} seconds)
3. After grace period, worker is eligible for idle checks
4. Every {poll} seconds, orchestrator checks:
   - Does worker have running tasks? → Keep active
   - No tasks AND above MIN_ACTIVE_GPUS ({min_gpus})? → Terminate
   - No tasks BUT at/below minimum? → Keep active
5. Termination is immediate once marked

Note: If tasks are queued, at least 1 worker is kept regardless of MIN_ACTIVE_GPUS
""")
        
        min_time = grace
        max_time = grace + poll
        print(f"⏱️  Idle Termination Time Range: {min_time/60:.1f} - {max_time/60:.1f} minutes")
        print(f"   (Depends on when orchestrator cycle runs after grace period expires)")
    
    print("\n" + "=" * 80)








