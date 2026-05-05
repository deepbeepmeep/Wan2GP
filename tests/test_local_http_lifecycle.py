from types import SimpleNamespace

from source.runtime.worker.server import _is_local_worker_mode


def _clear_local_worker_mode_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("WORKER_DB_CLIENT_AUTH_MODE", raising=False)


def test_is_local_worker_mode_true_with_pat_only(monkeypatch):
    _clear_local_worker_mode_env(monkeypatch)

    assert _is_local_worker_mode(SimpleNamespace(), access_token="pat-token") is True


def test_is_local_worker_mode_false_when_service_role_key_set(monkeypatch):
    _clear_local_worker_mode_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")

    assert _is_local_worker_mode(SimpleNamespace(), access_token="pat-token") is False


def test_is_local_worker_mode_false_when_service_key_alias_set(monkeypatch):
    _clear_local_worker_mode_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service")

    assert _is_local_worker_mode(SimpleNamespace(), access_token="pat-token") is False


def test_is_local_worker_mode_false_with_anon_key_only(monkeypatch):
    _clear_local_worker_mode_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")

    assert _is_local_worker_mode(SimpleNamespace(supabase_anon_key="anon"), access_token=None) is False


def test_is_local_worker_mode_false_when_auth_mode_service(monkeypatch):
    _clear_local_worker_mode_env(monkeypatch)
    monkeypatch.setenv("WORKER_DB_CLIENT_AUTH_MODE", "service")

    assert _is_local_worker_mode(SimpleNamespace(), access_token="pat-token") is False
