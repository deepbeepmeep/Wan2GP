# wan2gp-model-tracker plugin
# Plugin package initialization

__version__ = '1.0.0'

# Lazy imports to avoid circular dependencies and startup issues
def get_plugin_class():
    from .plugin import WAN2GPPlugin
    return WAN2GPPlugin

def register_plugin(wan2gp_instance):
    """Register plugin with Wan2GP"""
    from .plugin import WAN2GPPlugin
    return WAN2GPPlugin(wan2gp_instance)

__all__ = ['get_plugin_class', 'register_plugin', '__version__']
