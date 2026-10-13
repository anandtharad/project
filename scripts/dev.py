"""Cross-platform developer commands (Windows, macOS, Linux).

No shell-specific syntax (`export`, `$env:`, `curl`) is needed: run every
command the same way on any OS:

    uv run python scripts/dev.py doctor        # check setup and .env
    uv run python scripts/dev.py unit          # offline unit tests
    uv run python scripts/dev.py test          # full suite (needs the database)
    uv run python scripts/dev.py groq-models   # list models your Groq key can use
    uv run python scripts/dev.py live-groq     # live Groq tests (small free-tier usage)
    uv run python scripts/dev.py laya          # run the self-hosted Laya decision server

Keys are read from the project's .env file (any encoding Windows tools
write) or from the environment, and are never printed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings, detect_env_file_encoding  # noqa: E402

# The database started by `docker compose up -d db`.
DEFAULT_TEST_DATABASE_URL = "postgresql+asyncpg://companion:companion@localhost:5432/companion"


def _mask(secret: object) -> str:
    value = secret.get_secret_value() if secret is not None else ""  # type: ignore[attr-defined]
    return f"set ({value[:4]}..., {len(value)} chars)" if value else "not set"


def _pytest(args: list[str], extra_env: dict[str, str] | None = None) -> int:
    env = {**os.environ, **(extra_env or {})}
    # Fixed command: this interpreter running pytest with our own arguments.
    return subprocess.call([sys.executable, "-m", "pytest", *args], cwd=ROOT, env=env)  # noqa: S603


def _database_reachable(url: str) -> bool:
    from sqlalchemy import make_url

    parsed = make_url(url)
    try:
        with socket.create_connection((parsed.host or "localhost", parsed.port or 5432), timeout=2):
            return True
    except OSError:
        return False


def doctor(_: argparse.Namespace) -> int:
    ok = True
    print(
        f"Python      {sys.version.split()[0]}",
        "(ok)" if sys.version_info >= (3, 12) else "(need 3.12+)",
    )
    ok &= sys.version_info >= (3, 12)
    env_file = ROOT / ".env"
    if env_file.exists():
        encoding = detect_env_file_encoding(env_file)
        print(f".env        found, encoding {encoding}")
        if encoding == "utf-16":
            # The app reads it, but docker compose cannot parse a UTF-16 .env.
            print("  warning   docker compose cannot read a UTF-16 .env. Convert it to UTF-8 with:")
            print('            $c = Get-Content .env; [IO.File]::WriteAllLines("$PWD\\.env", $c)')
            ok = False
    else:
        print(".env        not found (copy .env.example to .env)")
    try:
        settings = Settings()
    except Exception as exc:  # report configuration problems plainly
        print(f"Settings    ERROR: {type(exc).__name__}: {exc}")
        return 1
    print(f"Provider    {settings.generation_provider}")
    for name in ("groq_api_key", "openai_api_key", "gemini_api_key", "anthropic_api_key"):
        print(f"  {name.upper():<20} {_mask(getattr(settings, name))}")
    try:
        from app.llm.providers.registry import ProviderConfigError, resolve_provider

        resolved = resolve_provider(settings)
        limits = f"effort {resolved.effort}, max output {resolved.max_output_tokens}"
        print(f"  model     {resolved.model}  ({limits})")
        for warning in resolved.warnings:
            print(f"  warning   {warning}")
    except ProviderConfigError as exc:
        print(f"  config    {exc}")
        ok = False
    for module, extra in (
        ("groq", "groq"),
        ("openai", "openai"),
        ("google.genai", "gemini"),
        ("anthropic", "anthropic"),
    ):
        try:
            __import__(module)
            status = "installed"
        except ImportError:
            status = f"missing (uv sync --extra {extra})"
        print(f"SDK {module:<14} {status}")
    print(
        "Docker      ", "found" if shutil.which("docker") else "not found (needed for the database)"
    )
    url = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL)
    print(
        "Database    ",
        "reachable" if _database_reachable(url) else "not reachable (run: docker compose up -d db)",
    )
    return 0 if ok else 1


def unit(args: argparse.Namespace) -> int:
    return _pytest(["tests/unit", "-q", *args.pytest_args])


def test(args: argparse.Namespace) -> int:
    url = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL)
    if not _database_reachable(url):
        print("Database not reachable. Start it with:  docker compose up -d db")
        return 1
    # RUN_LIVE_LLM_TESTS=0: a $env: value left over in PowerShell must not add live calls.
    return _pytest(["-q", *args.pytest_args], {"TEST_DATABASE_URL": url, "RUN_LIVE_LLM_TESTS": "0"})


def groq_models(_: argparse.Namespace) -> int:
    import groq

    secret = Settings().groq_api_key
    key = os.environ.get("GROQ_API_KEY") or (secret.get_secret_value() if secret else None)
    if not key:
        print("GROQ_API_KEY is not set (add it to .env)")
        return 1
    try:
        models = groq.Groq(api_key=key, max_retries=1).models.list().data
    except groq.AuthenticationError:
        print("Groq rejected the key (401). Check GROQ_API_KEY.")
        return 1
    except groq.APIConnectionError:
        print("Could not reach api.groq.com (network or firewall).")
        return 1
    for model in sorted(models, key=lambda m: m.id):
        context = getattr(model, "context_window", None)
        print(f"{model.id:<45} {getattr(model, 'owned_by', ''):<20} context={context}")
    return 0


LAYA_PACKAGE = "laya[serve]==0.4.6"


def laya(args: argparse.Namespace) -> int:
    """Run the pinned Laya server on localhost with the settings the app uses."""
    from urllib.parse import urlsplit

    settings = Settings()
    if settings.laya_api_key is None:
        print("Set LAYA_API_KEY in .env first (any long random string).")
        return 1
    port = urlsplit(settings.laya_base_url).port or 8100
    env = {
        **os.environ,
        "LAYA_HOST": "127.0.0.1",
        "LAYA_PORT": str(port),
        "LAYA_JEV_STRICT": "1",
        "LAYA_MODELS": settings.laya_model,
        "LAYA_PRELOAD": "1",
        "LAYA_API_KEY": settings.laya_api_key.get_secret_value(),
        "USE_TF": "0",
    }
    command = ["uv", "tool", "run", "--python", "3.12", "--from", LAYA_PACKAGE]
    if sys.platform.startswith("linux"):
        # The default Linux torch wheel bundles CUDA; use the CPU build.
        command += [
            "--with",
            "torch",
            "--index",
            "https://download.pytorch.org/whl/cpu",
            "--index-strategy",
            "unsafe-best-match",
        ]
    print(f"Starting {LAYA_PACKAGE} on http://127.0.0.1:{port} (first start downloads ~800 MB)")
    # Fixed command: uv running the pinned laya-serve entry point.
    return subprocess.call([*command, "laya-serve"], env=env)  # noqa: S603


def live_groq(args: argparse.Namespace) -> int:
    extra = {"RUN_LIVE_LLM_TESTS": "1"}
    if args.models:
        extra["LIVE_GROQ_MODELS"] = args.models
    return _pytest(["tests/live/test_groq_live.py", "-v", *args.pytest_args], extra)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check setup, .env and configuration").set_defaults(
        func=doctor
    )
    for name, func, text in (
        ("unit", unit, "offline unit tests"),
        ("test", test, "full test suite (needs the database)"),
    ):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("pytest_args", nargs="*", help="extra pytest arguments")
        sub.set_defaults(func=func)
    commands.add_parser("groq-models", help="list models available to your Groq key").set_defaults(
        func=groq_models
    )
    commands.add_parser("laya", help="run the self-hosted Laya decision server").set_defaults(
        func=laya
    )
    live = commands.add_parser("live-groq", help="live Groq tests (2 calls per model)")
    live.add_argument(
        "--models", help="comma-separated Groq model ids (default: 3 free-tier models)"
    )
    live.add_argument("pytest_args", nargs="*", help="extra pytest arguments")
    live.set_defaults(func=live_groq)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
