#!/usr/bin/env python3
"""
Test script to verify the plugin loads correctly
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("Testing plugin imports...")

try:
    from plugin import WAN2GPPlugin, register_plugin
    print("✓ Successfully imported WAN2GPPlugin and register_plugin")
except Exception as e:
    print(f"✗ Error importing plugin: {e}")
    sys.exit(1)

try:
    from performance_db import PerformanceDatabase
    print("✓ Successfully imported PerformanceDatabase")
except Exception as e:
    print(f"✗ Error importing performance_db: {e}")
    sys.exit(1)

try:
    from model_analyzer import ModelAnalyzer
    print("✓ Successfully imported ModelAnalyzer")
except Exception as e:
    print(f"✗ Error importing model_analyzer: {e}")
    sys.exit(1)

print("\nTesting plugin instantiation with mock instance...")

class MockWan2GP:
    """Mock Wan2GP instance for testing"""
    def __init__(self):
        self.models_def = {
            "test_model": {
                "name": "Test Model",
                "description": "Test model for plugin verification",
                "URLs": ["https://example.com/test.safetensors"]
            }
        }
        self.files_locator = None

    def get_local_model_filename(self, url):
        return url.split('/')[-1]

try:
    mock_wan2gp = MockWan2GP()
    plugin = WAN2GPPlugin(mock_wan2gp)
    print("✓ Successfully instantiated plugin")
except Exception as e:
    print(f"✗ Error instantiating plugin: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    info = plugin.get_plugin_info()
    print(f"✓ Plugin info: {info['name']} v{info['version']}")
except Exception as e:
    print(f"✗ Error getting plugin info: {e}")
    sys.exit(1)

print("\n✓ All tests passed! Plugin is ready to use.")
print("\nManual Installation Instructions:")
print("1. Copy this entire directory to: <wan2gp-path>/plugins/wan2gp-model-tracker/")
print("2. Restart Wan2GP")
print("3. Enable the plugin in the Plugins tab")
print("4. Look for the '📊 Models' tab")
