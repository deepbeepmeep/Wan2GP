def magi_compile(*args, **kwargs):
    def decorator(obj):
        return obj

    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return decorator
