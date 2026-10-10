"""MCP OAuth 2.1 authorization server using the MCP SDK's protocol handlers."""
import re
import secrets
import time
from urllib.parse import urlsplit

from mcp.server.auth.handlers.token import TokenHandler
from mcp.server.auth.handlers.authorize import AuthorizationHandler
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.middleware.client_auth import AuthenticationError, ClientAuthenticator
from mcp.server.auth.provider import AccessToken, AuthorizationCode, AuthorizeError, RefreshToken, TokenError, construct_redirect_uri
from mcp.server.auth.routes import cors_middleware, create_auth_routes, create_protected_resource_routes
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyHttpUrl, ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route

from . import password
from .web import PAGE_HEADERS, PasswordGate, login_page, same_origin, small_form

SCOPE = "wangp"


class GrantAccessToken(AccessToken):
    grant_id: str


class GrantRefreshToken(RefreshToken):
    grant_id: str


class ExpiringStore:
    def __init__(self, limit=4096):
        self.values = {}
        self.limit = limit

    def put(self, key, value, expires):
        now = time.time()
        self.values = {k: item for k, item in self.values.items() if item[1] > now}
        if len(self.values) >= self.limit:
            raise ValueError("Authorization capacity reached. Try again later.")
        self.values[key] = value, expires

    def get(self, key):
        item = self.values.get(key)
        if item is not None:
            if item[1] > time.time():
                return item[0]
            self.values.pop(key, None)
        return None

    def pop(self, key):
        value = self.get(key)
        self.values.pop(key, None)
        return value


class MCPAuthorization:
    def __init__(self, passphrase, issuer, resource):
        self.gate = PasswordGate(passphrase)
        self.issuer, self.resource = issuer, resource
        self.clients = ExpiringStore(512)
        self.pending = ExpiringStore(512)
        self.codes = ExpiringStore()
        self.access = ExpiringStore()
        self.refresh = ExpiringStore()
        self.grants = ExpiringStore()

    async def get_client(self, client_id):
        return self.clients.get(client_id)

    async def register_client(self, client_info):
        self.clients.put(client_info.client_id, client_info, time.time() + 600)

    async def authorize(self, client, params):
        if params.resource != self.resource:
            raise AuthorizeError("invalid_request", "resource must be the advertised MCP resource URL.")
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", params.code_challenge):
            raise AuthorizeError("invalid_request", "An S256 PKCE challenge is required.")
        if params.scopes != [SCOPE]:
            raise AuthorizeError("invalid_scope", "The wangp scope is required.")
        pending_id = secrets.token_urlsafe(32)
        try:
            self.pending.put(pending_id, {"client": client, "params": params, "csrf": secrets.token_urlsafe(32), "browser": None}, time.time() + 600)
        except ValueError as exc:
            raise AuthorizeError("temporarily_unavailable", str(exc)) from exc
        return self.issuer + "/oauth/approve?request=" + pending_id

    async def consent(self, request):
        pending_id = request.query_params.get("request", "")
        pending = self.pending.get(pending_id)
        if pending is None:
            return JSONResponse({"detail": "Authorization request expired. Reconnect your MCP client."}, status_code=400, headers=PAGE_HEADERS)
        client, params = pending["client"], pending["params"]
        error, status, retry = "", 200, 0
        browser_cookie = "wangp_oauth_" + pending_id
        browser = request.cookies.get(browser_cookie)
        if request.method == "POST":
            try:
                form = await small_form(request)
            except (ValueError, UnicodeError):
                return JSONResponse({"detail": "Invalid authorization request."}, status_code=400)
            if not same_origin(request) or browser is None or pending["browser"] is None or not secrets.compare_digest(browser.encode("utf-8"), pending["browser"].encode("utf-8")) or not secrets.compare_digest(form.get("csrf", "").encode("utf-8"), pending["csrf"].encode("utf-8")):
                return JSONResponse({"detail": "Authorization form verification failed."}, status_code=403)
            valid, retry = await run_in_threadpool(self.gate.check, form.get("password", ""))
            if valid:
                if self.pending.pop(pending_id) is None:
                    return JSONResponse({"detail": "Authorization request already used or expired."}, status_code=400)
                self.clients.put(client.client_id, client, time.time() + 30 * 86400)
                code = AuthorizationCode(code=secrets.token_urlsafe(32), client_id=client.client_id, scopes=params.scopes, expires_at=time.time() + 120, code_challenge=params.code_challenge, redirect_uri=params.redirect_uri, redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly, resource=params.resource)
                self.codes.put(code.code, code, code.expires_at)
                response = RedirectResponse(construct_redirect_uri(str(params.redirect_uri), code=code.code, state=params.state, iss=self.issuer + "/"), status_code=303, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})
                response.delete_cookie(browser_cookie, path="/oauth/approve")
                return response
            error = f"Try again in {retry} seconds." if retry else "Incorrect MCP passphrase."
            status = 429 if retry else 401
        else:
            browser = secrets.token_urlsafe(32)
            pending["browser"] = browser
        description = f"{client.client_name or client.client_id} requests access to WanGP tools and media, including generation and the filesystem permissions enabled on this server. Approval returns to {params.redirect_uri}. Enter the separate MCP passphrase only if you initiated this connection."
        response = login_page("Authorize MCP Access", description, "/oauth/approve?request=" + pending_id, {"csrf": pending["csrf"]}, error=error, status=status, retry=retry, button="Authorize Client")
        response.set_cookie(browser_cookie, browser, secure=request.url.scheme == "https", httponly=True, samesite="lax", path="/oauth/approve", max_age=600)
        return response

    async def load_authorization_code(self, client, authorization_code):
        return self.codes.get(authorization_code)

    def issue_tokens(self, client_id, scopes, grant_id, expires):
        access = GrantAccessToken(token=secrets.token_urlsafe(32), client_id=client_id, scopes=scopes, expires_at=min(int(time.time()) + 3600, expires), resource=self.resource, grant_id=grant_id)
        refresh = GrantRefreshToken(token=secrets.token_urlsafe(32), client_id=client_id, scopes=scopes, expires_at=expires, grant_id=grant_id)
        self.access.put(access.token, access, access.expires_at)
        self.refresh.put(refresh.token, refresh, expires)
        self.grants.put(grant_id, (access.token, refresh.token), expires)
        return OAuthToken(access_token=access.token, token_type="Bearer", expires_in=access.expires_at - int(time.time()), refresh_token=refresh.token, scope=" ".join(scopes))

    async def exchange_authorization_code(self, client, authorization_code):
        code = self.codes.pop(authorization_code.code)
        if code is None or code.client_id != client.client_id:
            raise TokenError("invalid_grant", "Authorization code already used or expired.")
        return self.issue_tokens(client.client_id, code.scopes, secrets.token_urlsafe(32), int(time.time()) + 7 * 86400)

    async def load_refresh_token(self, client, refresh_token):
        return self.refresh.get(refresh_token)

    async def exchange_refresh_token(self, client, refresh_token, scopes):
        grant = self.grants.get(refresh_token.grant_id)
        if grant is None:
            raise TokenError("invalid_grant", "Authorization has expired or was revoked.")
        if grant[1] != refresh_token.token:
            self.grants.pop(refresh_token.grant_id)
            raise TokenError("invalid_grant", "Refresh token reuse detected. Reconnect the MCP client.")
        return self.issue_tokens(client.client_id, scopes, refresh_token.grant_id, refresh_token.expires_at)

    async def load_access_token(self, token):
        access = self.access.get(token)
        if access is not None:
            grant = self.grants.get(access.grant_id)
            if grant is not None and grant[0] == access.token:
                return access
        return None

    async def verify_token(self, token):
        return await self.load_access_token(token)

    async def revoke_token(self, token):
        self.grants.pop(token.grant_id)

    async def revoke(self, request):
        form = await request.form()
        if not form.get("client_id") or not form.get("token"):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        try:
            client = await ClientAuthenticator(self).authenticate_request(request)
        except AuthenticationError:
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        # The SDK's revocation model requires client_secret even for public
        # clients. Keep its authenticator, but accept the standard optional field.
        token = self.access.get(form["token"]) or self.refresh.get(form["token"])
        if token is not None and token.client_id == client.client_id:
            await self.revoke_token(token)
        return Response(status_code=200, headers={"Cache-Control": "no-store"})

    async def authorization_endpoint(self, request):
        response = await AuthorizationHandler(self).handle(request)
        if response.status_code == 302 and not response.headers['location'].startswith(self.issuer + "/oauth/approve?"):
            response.headers['location'] = construct_redirect_uri(response.headers['location'], iss=self.issuer + "/")
        return response

    async def register(self, request):
        try:
            metadata = OAuthClientMetadata.model_validate(await request.json())
            if metadata.token_endpoint_auth_method not in {"none", "client_secret_post", "client_secret_basic"}:
                raise ValueError("Supported client authentication methods: none, client_secret_post, client_secret_basic.")
            if "authorization_code" not in metadata.grant_types or not set(metadata.grant_types) <= {"authorization_code", "refresh_token"} or metadata.response_types != ["code"]:
                raise ValueError("Only authorization_code and optional refresh_token grants with response type code are supported.")
            if metadata.scope is not None and metadata.scope.split() != [SCOPE]:
                raise ValueError("The supported scope is wangp.")
            for redirect in metadata.redirect_uris:
                parsed = urlsplit(str(redirect))
                if parsed.fragment or parsed.username or parsed.password or not (parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"})):
                    raise ValueError("Redirects must use HTTPS or a loopback HTTP address, without credentials or fragments.")
            client = OAuthClientInformationFull(**metadata.model_dump(exclude={"scope"}), scope=SCOPE, client_id=secrets.token_urlsafe(24), client_id_issued_at=int(time.time()), client_secret=secrets.token_urlsafe(32) if metadata.token_endpoint_auth_method != "none" else None)
            await self.register_client(client)
            return JSONResponse(client.model_dump(mode="json", exclude_none=True), status_code=201, headers={"Cache-Control": "no-store"})
        except (ValueError, ValidationError) as exc:
            return JSONResponse({"error": "invalid_client_metadata", "error_description": str(exc)}, status_code=400, headers={"Cache-Control": "no-store"})

    async def token(self, request):
        # The installed SDK parses resource but does not validate it during token
        # exchange. All grants here are bound to exactly this MCP resource.
        form = await request.form()
        if form.get("resource") != self.resource:
            return JSONResponse({"error": "invalid_target", "error_description": "resource must match the advertised MCP resource URL."}, status_code=400, headers={"Cache-Control": "no-store"})
        if form.get("grant_type") == "authorization_code" and not re.fullmatch(r"[A-Za-z0-9._~-]{43,128}", form.get("code_verifier", "")):
            return JSONResponse({"error": "invalid_grant", "error_description": "Invalid PKCE verifier."}, status_code=400, headers={"Cache-Control": "no-store"})
        return await TokenHandler(self, ClientAuthenticator(self)).handle(request)

    def wrap(self, app):
        options = ClientRegistrationOptions(enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE])
        routes = create_auth_routes(self, AnyHttpUrl(self.issuer), client_registration_options=options, revocation_options=RevocationOptions(enabled=True))
        for route in routes:
            if route.path == "/register":
                route.app = cors_middleware(self.register, ["POST", "OPTIONS"])
            elif route.path == "/token":
                route.app = cors_middleware(self.token, ["POST", "OPTIONS"])
            elif route.path == "/revoke":
                route.app = cors_middleware(self.revoke, ["POST", "OPTIONS"])
            elif route.path == "/authorize":
                from starlette.routing import request_response
                route.app = request_response(self.authorization_endpoint)
            elif route.path == "/.well-known/oauth-authorization-server":
                from mcp.server.auth.routes import build_metadata
                metadata = build_metadata(AnyHttpUrl(self.issuer), None, options, RevocationOptions(enabled=True)).model_dump(mode="json", exclude_none=True)
                metadata.update(token_endpoint_auth_methods_supported=["none", "client_secret_post", "client_secret_basic"], revocation_endpoint_auth_methods_supported=["none", "client_secret_post", "client_secret_basic"], authorization_response_iss_parameter_supported=True)

                async def discovery(request):
                    return JSONResponse(metadata)

                route.app = cors_middleware(discovery, ["GET", "OPTIONS"])
        routes.extend(create_protected_resource_routes(AnyHttpUrl(self.resource), [AnyHttpUrl(self.issuer)], [SCOPE]))
        # Path-aware RFC 9728 discovery, as well as the root discovery endpoint.
        resource_path = urlsplit(self.resource).path
        routes.append(Route("/.well-known/oauth-protected-resource" + resource_path, routes[-1].endpoint, methods=["GET", "OPTIONS"]))
        routes.append(Route("/oauth/approve", self.consent, methods=["GET", "POST"]))
        protected = RequireAuthMiddleware(app, [SCOPE], AnyHttpUrl(self.issuer + "/.well-known/oauth-protected-resource" + resource_path))
        protected = AuthenticationMiddleware(protected, backend=BearerAuthBackend(self))
        routes.append(Mount("/", protected))
        return OAuthRequestLimits(Starlette(routes=routes, lifespan=app.router.lifespan_context), https=self.issuer.startswith("https:"))


class OAuthRequestLimits:
    """Bound untrusted OAuth requests without limiting authenticated media uploads."""
    def __init__(self, app, *, https):
        self.app, self.https = app, https

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.https and scope["scheme"] != "https":
            return await JSONResponse({"error": "invalid_request", "error_description": "Use the configured HTTPS origin."}, status_code=403)(scope, receive, send)
        if scope["type"] == "http" and scope["path"] in {"/authorize", "/token", "/register", "/revoke", "/oauth/approve"}:
            if len(scope["query_string"]) > 16384:
                return await JSONResponse({"error": "invalid_request"}, status_code=414)(scope, receive, send)
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > 16384:
                    return await JSONResponse({"error": "invalid_request"}, status_code=413)(scope, receive, send)
                if not message.get("more_body"):
                    break

            async def replay():
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            return await self.app(scope, replay, send)
        return await self.app(scope, receive, send)


def mcp_authorization(args, transport):
    if args.auth or args.auth_password is not None:
        raise ValueError("MCP uses separate OAuth authentication. Use --mcp-auth and --mcp-auth-password.")
    if transport == "stdio" and (args.mcp_auth or args.mcp_auth_password is not None or args.mcp_auth_url is not None):
        raise ValueError("MCP OAuth requires an HTTP transport; stdio does not use web authentication.")
    if args.mcp_auth_url is not None and not args.mcp_auth:
        raise ValueError("--mcp-auth-url requires --mcp-auth.")
    if args.mcp_auth and args.mcp_auth_url is None:
        raise ValueError("--mcp-auth requires --mcp-auth-url with the public server origin.")
    if args.mcp_auth:
        from importlib.metadata import version
        from packaging.version import Version
        if Version(version("mcp")) < Version("1.30.0"):
            raise RuntimeError("MCP OAuth requires the updated dependencies. Run: python -m pip install -r requirements.txt")
    passphrase = password(args, mcp=True)
    if passphrase is None:
        return None
    issuer = args.mcp_auth_url.rstrip("/")
    parsed = urlsplit(issuer)
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path or not (parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"})):
        raise ValueError("--mcp-auth-url must be an HTTPS origin (loopback HTTP is allowed), without a path, query or credentials.")
    issuer = str(AnyHttpUrl(issuer)).rstrip("/")
    return MCPAuthorization(passphrase, issuer, issuer + ("/sse" if transport == "sse" else "/mcp"))


def transport_security(issuer):
    from mcp.server.transport_security import TransportSecuritySettings
    return TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=[urlsplit(issuer).netloc, "127.0.0.1:*", "localhost:*", "[::1]:*"], allowed_origins=[issuer])
