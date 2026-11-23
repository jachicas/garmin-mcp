"""OAuth provider must survive a server restart without forcing a new login."""

import asyncio

import pytest
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull

from garmin_mcp.auth import SimpleOAuthProvider

ISSUER = "https://garmin.example"
SECRET = "x" * 48
REDIRECT = "https://claude.ai/api/mcp/auth_callback"


def provider() -> SimpleOAuthProvider:
    return SimpleOAuthProvider(mcp_password="pw", jwt_secret=SECRET, issuer_url=ISSUER)


def run(coro):
    return asyncio.run(coro)


def login(p: SimpleOAuthProvider, client: OAuthClientInformationFull):
    params = AuthorizationParams(
        state="s",
        scopes=["mcp"],
        code_challenge="c" * 43,
        redirect_uri=REDIRECT,
        redirect_uri_provided_explicitly=True,
    )
    url = run(p.authorize(client, params))
    state = url.split("state=")[1]
    redirect = run(p.complete_login(state, "pw"))
    code_value = redirect.split("code=")[1].split("&")[0]
    code = run(p.load_authorization_code(client, code_value))
    return run(p.exchange_authorization_code(client, code))


def registered_client() -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id="claude-123",
        client_secret="sekret",
        redirect_uris=[REDIRECT],
        token_endpoint_auth_method="client_secret_post",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="mcp",
    )


def test_refresh_works_after_restart():
    p1 = provider()
    client = registered_client()
    run(p1.register_client(client))
    tokens = login(p1, client)

    # Fresh process: in-memory registry is empty.
    p2 = provider()
    restored = run(p2.get_client("claude-123"))
    assert restored is not None
    assert restored.token_endpoint_auth_method == "none"
    assert [str(u) for u in restored.redirect_uris] == [REDIRECT, "https://claude.com/api/mcp/auth_callback"]

    rt = run(p2.load_refresh_token(restored, tokens.refresh_token))
    assert rt is not None and rt.client_id == "claude-123"
    new = run(p2.exchange_refresh_token(restored, rt, []))
    assert run(p2.load_access_token(new.access_token)) is not None
    # The new refresh token keeps working too.
    assert run(p2.load_refresh_token(restored, new.refresh_token)) is not None


def test_refresh_token_rejected_for_other_client_or_secret():
    p = provider()
    client = registered_client()
    run(p.register_client(client))
    tokens = login(p, client)
    other = run(p.get_client("someone-else"))
    assert run(p.load_refresh_token(other, tokens.refresh_token)) is None

    forged = SimpleOAuthProvider(mcp_password="pw", jwt_secret="y" * 48, issuer_url=ISSUER)
    assert run(forged.load_refresh_token(client, tokens.refresh_token)) is None


def test_refresh_token_is_not_an_access_token():
    p = provider()
    client = registered_client()
    run(p.register_client(client))
    tokens = login(p, client)
    assert run(p.load_access_token(tokens.refresh_token)) is None


def test_revoked_refresh_token_is_rejected():
    p = provider()
    client = registered_client()
    run(p.register_client(client))
    tokens = login(p, client)
    rt = run(p.load_refresh_token(client, tokens.refresh_token))
    run(p.revoke_token(rt))
    assert run(p.load_refresh_token(client, tokens.refresh_token)) is None


def test_new_login_still_needs_password():
    p = provider()
    client = run(p.get_client("restored"))
    params = AuthorizationParams(
        state="s", scopes=["mcp"], code_challenge="c" * 43,
        redirect_uri=REDIRECT, redirect_uri_provided_explicitly=True,
    )
    url = run(p.authorize(client, params))
    with pytest.raises(Exception):
        run(p.complete_login(url.split("state=")[1], "wrong"))
