# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Notify the shared UI stream after generation control state changes."""


def generation_state_changed(gen):
    callback = gen.get('state_change_callback')
    if callback is not None:
        callback()


def set_generation_state(gen, **values):
    changed = any(gen.get(key) != value for key, value in values.items())
    gen.update(values)
    if changed:
        generation_state_changed(gen)


def pause_state(gen):
    status = gen.get('process_status') or ''
    return (status == 'request:pause' or status.startswith('request:deepy_pause_'),
            status == 'process:pause' or status.startswith('process:deepy_pause_'))
