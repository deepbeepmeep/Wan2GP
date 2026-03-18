"""Worker investigation command."""

from debug.client import DebugClient
from debug.formatting.worker_detail_formatter import format_worker


def run(client: DebugClient, worker_id: str | None = None, options: dict | None = None):
    """Handle 'debug.py worker <id>' command."""
    options = options or {}
    worker_id = worker_id or options.get("worker_id")
    try:
        if not worker_id:
            print("❌ Error investigating worker: missing worker_id")
            return False

        hours = options.get('hours', 24)
        startup = options.get('startup', False)
        check_logging = options.get('check_logging', False)
        check_disk = options.get('check_disk', False)
        
        # Check disk space via SSH
        if check_disk:
            result = client.check_worker_disk_space(worker_id)
            print("=" * 80)
            print(f"💾 DISK SPACE CHECK: {worker_id}")
            print("=" * 80)
            
            if result.get('available'):
                print(f"\n✅ SSH successful (RunPod: {result.get('runpod_id')})\n")
                print(result.get('disk_info', 'No output'))
                
                issues = result.get('issues', [])
                if issues:
                    print("\n" + "=" * 40)
                    print("⚠️  ISSUES DETECTED:")
                    for issue in issues:
                        print(f"   {issue}")
                else:
                    print("\n✅ No disk space issues detected")
            else:
                print(f"\n❌ Cannot check disk space: {result.get('error')}")
                print("\n💡 Worker may be terminated or SSH unavailable")
            print()
            return True
        
        # Check if worker is logging
        if check_logging:
            result = client.check_worker_logging(worker_id)
            print("=" * 80)
            print(f"🔍 WORKER LOGGING CHECK: {worker_id}")
            print("=" * 80)
            
            if result['is_logging']:
                print(f"\n✅ Worker IS logging! ({result['log_count']} recent logs)")
                print(f"\nMost recent logs:")
                print("-" * 80)
                for log in result['recent_logs']:
                    timestamp = log['timestamp'][-12:-4]
                    level = log['log_level']
                    message = log['message'][:70]
                    print(f"[{level:7}] {timestamp} | {message}")
            else:
                print(f"\n❌ Worker is NOT logging yet")
                print("   This means worker.py has not started or crashed during initialization")
                print("\n💡 Next steps:")
                print(f"   1. Check worker status: debug.py worker {worker_id}")
                print(f"   2. Check startup logs: debug.py worker {worker_id} --startup")
                print("   3. Wait if worker is < 10 minutes old (might be installing dependencies)")
            print()
            return True
        
        # Get worker info
        info = client.get_worker_info(worker_id, hours=hours, startup=startup)
        
        format_type = options.get('format', 'text')
        logs_only = options.get('logs_only', False)
        
        if startup and format_type == 'text':
            # Special formatting for startup mode
            print("=" * 80)
            print(f"🚀 WORKER STARTUP LOGS: {worker_id}")
            print("=" * 80)
            print(f"\nFound {len(info.logs)} startup-related log entries\n")
            
            if not info.logs:
                print("⚠️  No startup logs found")
                print("   Worker may have been created before logging was implemented")
                print("   or is still being provisioned")
            else:
                for log in info.logs[:100]:  # Show first 100 startup logs
                    timestamp = log['timestamp'][11:19]
                    level = log['log_level']
                    message = log['message']
                    
                    level_symbol = {
                        'ERROR': '❌',
                        'WARNING': '⚠️',
                        'INFO': 'ℹ️',
                        'DEBUG': '🔍'
                    }.get(level, '  ')
                    
                    print(f"[{timestamp}] {level_symbol} {message}")
                
                # Check for common issues
                all_messages = ' '.join([log['message'] for log in info.logs])
                if 'ModuleNotFoundError' in all_messages:
                    print("\n⚠️  ISSUE DETECTED: Missing Python module")
                    print("   Worker crashed due to missing dependencies")
                elif 'died immediately' in all_messages:
                    print("\n❌ ISSUE DETECTED: Worker process died immediately")
                elif 'still running' in all_messages:
                    print("\n✅ Worker process started successfully")
        else:
            output = format_worker(info, format_type, logs_only)
            print(output)
        return True
        
    except Exception as e:
        print(f"❌ Error investigating worker: {e}")
        import traceback
        if options.get('debug'):
            print(traceback.format_exc())
        return False
