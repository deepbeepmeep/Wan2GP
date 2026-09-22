"""Shared launch options for web authentication, MCP OAuth and TLS."""
import argparse
import os
import secrets
from urllib.parse import urlsplit


def parse_public_url(value):
    from pydantic import AnyHttpUrl

    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in {'', '/'}
                or any(char.isspace() or ord(char) < 32 for char in value)
                or any(char in value for char in ('?', '#', '\\', '*'))):
            raise ValueError()
        url = AnyHttpUrl(value)
        if not 1 <= url.port <= 65535:
            raise ValueError()
        return str(url).rstrip('/')
    except ValueError as error:
        raise ValueError('--public-url must be an HTTP(S) origin, e.g. https://wangp.example.com, without a path, query, fragment or credentials.') from error


def add_arguments(parser):
    add = parser.add_argument
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--auth", action="store_true", help="Require a password for all Gradio and Deepy web access")
    auth.add_argument("--no-auth", action="store_false", dest="auth", help="Disable web authentication (default)")
    add("--auth-password", default=None, help="Web passphrase (requires --auth; otherwise WANGP_AUTH_PASSWORD or a generated password)")
    add("--public-url", type=parse_public_url, default=None, metavar="ORIGIN", help="Restrict Gradio/Deepy browser requests to this exact origin, e.g. https://wangp.example.com (default: same host/port, HTTP or HTTPS)")
    add("--mcp-auth", action="store_true", help="Require OAuth authorization for network MCP")
    add("--mcp-auth-password", default=None, help="Separate MCP approval passphrase (requires --mcp-auth; otherwise WANGP_MCP_AUTH_PASSWORD or a generated password)")
    add("--mcp-auth-url", default=None, help="Public MCP server origin, e.g. https://wangp.example.com:7866 (requires --mcp-auth)")
    add("--https-port", type=int, default=None, help="Serve HTTPS on this port and redirect the main HTTP port to HTTPS")
    add("--ssl-certfile", default=None, help="TLS certificate PEM file (otherwise WANGP_SSL_CERT)")
    add("--ssl-keyfile", default=None, help="TLS private key PEM file (otherwise WANGP_SSL_KEY)")


def forwarded_options(args):
    # wgp.py forwards launch flags through cli_arg; the module entry point already
    # has these attributes. argparse preserves explicit namespace values.
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    parser.parse_known_args(args.cli_arg, namespace=args)
    return args


def password(args, *, mcp=False):
    enabled = args.mcp_auth if mcp else args.auth
    supplied = args.mcp_auth_password if mcp else args.auth_password
    flag = "--mcp-auth" if mcp else "--auth"
    if supplied is not None and not enabled:
        raise ValueError(f"{flag}-password requires {flag}.")
    if not enabled:
        return None
    value = supplied if supplied is not None else os.getenv("WANGP_MCP_AUTH_PASSWORD" if mcp else "WANGP_AUTH_PASSWORD")
    if value is None:
        value = secrets.token_urlsafe(24)
        print(f"{'MCP approval' if mcp else 'WanGP web'} password: {value}")
    if not value.strip() or len(value.encode("utf-8")) > 1024:
        raise ValueError("The passphrase must be non-empty and at most 1024 UTF-8 bytes.")
    return value
