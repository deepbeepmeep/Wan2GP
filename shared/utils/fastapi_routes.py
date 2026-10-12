# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Iterate effective routes across eager and lazy FastAPI router inclusion."""


def iter_app_routes(app):
    for route in app.routes:
        if hasattr(route, 'effective_route_contexts'):
            # FastAPI 0.137+ keeps included routers nested. These contexts include
            # the complete path prefix and preserve the original endpoint.
            yield from route.effective_route_contexts()
        else:
            yield route
