"""Storage health check command."""

import os
from dotenv import load_dotenv
from debug.client import DebugClient


def run(client: DebugClient, options: dict):
    """Handle 'debug.py storage' command - check all storage volumes."""
    load_dotenv()
    success = True
    
    print("=" * 80)
    print("📦 STORAGE HEALTH CHECK")
    print("=" * 80)
    
    try:
        from gpu_orchestrator.runpod_client import create_runpod_client, get_network_volumes
        
        runpod_client = create_runpod_client()
        
        # Get all storage volumes from RunPod
        print("\n📊 RunPod Network Volumes:\n")
        volumes = get_network_volumes(runpod_client.api_key)
        
        if not volumes:
            print("   No network volumes found")
            return True
        
        for vol in volumes:
            name = vol.get('name', 'Unknown')
            vol_id = vol.get('id', 'Unknown')
            size_gb = vol.get('size', 0)
            dc = vol.get('dataCenter', {})
            dc_name = dc.get('name', 'Unknown')
            
            print(f"   📁 {name}")
            print(f"      ID: {vol_id}")
            print(f"      Size: {size_gb} GB")
            print(f"      Location: {dc_name}")
            print()
        
        # Get active workers to check actual usage
        print("\n🔍 Checking Actual Usage via Active Workers:\n")
        
        workers = client.supabase.table('workers').select('*').eq('status', 'active').execute()
        
        if not workers.data:
            print("   No active workers to check storage via SSH")
            print("   (Need an active worker to SSH and check actual disk usage)")
            return True
        
        # Group workers by storage volume
        workers_by_storage = {}
        for worker in workers.data:
            storage = worker.get('metadata', {}).get('storage_volume')
            if storage:
                if storage not in workers_by_storage:
                    workers_by_storage[storage] = []
                workers_by_storage[storage].append(worker)
        
        if not workers_by_storage:
            print("   No workers have storage_volume metadata")
            return True
        
        print(f"   Found workers on {len(workers_by_storage)} storage volume(s): {list(workers_by_storage.keys())}\n")
        
        # Check each storage volume
        for storage_name, storage_workers in workers_by_storage.items():
            print(f"   {'='*60}")
            print(f"   📦 {storage_name}")
            print(f"   {'='*60}")
            
            # Get volume ID
            volume_id = runpod_client._get_storage_volume_id(storage_name)
            if not volume_id:
                print(f"      ❌ Could not find volume ID")
                continue
            
            # Pick a worker with recent heartbeat
            check_worker = None
            for w in storage_workers:
                runpod_id = w.get('metadata', {}).get('runpod_id')
                if runpod_id:
                    check_worker = w
                    break
            
            if not check_worker:
                print(f"      ❌ No worker available to SSH")
                continue
            
            runpod_id = check_worker.get('metadata', {}).get('runpod_id')
            print(f"      Checking via worker: {check_worker['id']}")
            print(f"      RunPod ID: {runpod_id}")
            
            # Check storage health
            health = runpod_client.check_storage_health(
                storage_name=storage_name,
                volume_id=volume_id,
                active_runpod_id=runpod_id,
                min_free_gb=int(os.getenv('STORAGE_MIN_FREE_GB', '50')),
                max_percent_used=int(os.getenv('STORAGE_MAX_PERCENT_USED', '85'))
            )
            
            print()
            if health.get('error'):
                print(f"      ❌ Error: {health.get('message')}")
            else:
                status = "✅ HEALTHY" if health.get('healthy') else "❌ UNHEALTHY"
                print(f"      Status: {status}")
                print(f"      Total: {health.get('total_gb', 'N/A')} GB")
                print(f"      Used: {health.get('used_gb', 'N/A')} GB ({health.get('percent_used', 'N/A')}%)")
                print(f"      Free: {health.get('free_gb', 'N/A')} GB")
                print(f"      Message: {health.get('message')}")
                
                if health.get('needs_expansion'):
                    print()
                    print(f"      ⚠️  NEEDS EXPANSION!")
                    print(f"      Run: python scripts/debug.py storage --expand {storage_name}")
            
            print()
        
        # Show expansion option if requested
        expand_target = options.get('expand')
        if expand_target:
            print(f"\n🔧 Expanding storage '{expand_target}'...")
            
            volume_id = runpod_client._get_storage_volume_id(expand_target)
            if not volume_id:
                print(f"   ❌ Could not find volume ID for '{expand_target}'")
                return False
            
            # Get current size
            volumes = get_network_volumes(runpod_client.api_key)
            volume_info = next((v for v in volumes if v.get('name') == expand_target), None)
            
            if not volume_info:
                print(f"   ❌ Could not find volume info for '{expand_target}'")
                return False
            
            current_size = volume_info.get('size', 100)
            increment = int(os.getenv('STORAGE_EXPANSION_INCREMENT_GB', '50'))
            new_size = current_size + increment
            
            print(f"   Current size: {current_size} GB")
            print(f"   New size: {new_size} GB (+{increment} GB)")
            
            if runpod_client._expand_network_volume(volume_id, new_size):
                print(f"   ✅ Successfully expanded to {new_size} GB!")
            else:
                print(f"   ❌ Expansion failed for '{expand_target}'")
                success = False
        
    except (ImportError, ValueError, OSError) as e:
        print(f"❌ Error checking storage: {e}")
        import traceback
        if options.get('debug'):
            print(traceback.format_exc())
        success = False

    print()
    print("=" * 80)
    return success
