def retry_on_oom(operation, *args, **kwargs):
    kwargs.pop("debug", None)
    kwargs.pop("operation_name", None)
    return operation(*args, **kwargs)
