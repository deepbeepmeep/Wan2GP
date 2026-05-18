def magi_register_custom_op(*args, **kwargs):
    def decorator(fn):
        return fn

    return decorator
