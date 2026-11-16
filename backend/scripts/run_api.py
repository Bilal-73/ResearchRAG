#!/usr/bin/env python
"""Run the HTTP API.

    python scripts/run_api.py                    # http://127.0.0.1:8000, docs at /docs
    python scripts/run_api.py --reload           # auto-reload while developing
    python scripts/run_api.py --host 0.0.0.0 --port 8080

IMPORTANT (Windows): run this with the virtualenv's interpreter, not whatever
`python` resolves to on PATH:

    .venv\\Scripts\\python.exe scripts/run_api.py

Equivalent to:  uvicorn research_rag.api.app:app --app-dir src
"""
import argparse
import socket
import sys

from _bootstrap import get_settings  # noqa: F401 — sets sys.path, logging and UTF-8 output first

def _check_port(host: str, port: int) -> None:
    """Uvicorn's own 'address already in use' is easy to miss in a wall of logs,
    and the usual cause is a server the user forgot was running."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host if host != "0.0.0.0" else "", port))
    except OSError:
        print(f"\n  Port {port} is already in use — another server is probably still running.",
              file=sys.stderr)
        print(f"  Use a different port:  scripts/run_api.py --port {port + 1}", file=sys.stderr)
        print("  Or find and stop the process holding it:", file=sys.stderr)
        print(f'      netstat -ano | findstr ":{port}"   then   taskkill /F /PID <pid>', file=sys.stderr)
        sys.exit(1)
    finally:
        probe.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Each worker builds its own agent (and loads the cross-encoder), so "
                             "memory scales with this. Conversation memory is per-process unless "
                             "CHECKPOINT_BACKEND=sqlite.")
    args = parser.parse_args()

    _check_port(args.host, args.port)

    import uvicorn

    print(f"\n  Docs:   http://{args.host}:{args.port}/docs")
    print(f"  Health: http://{args.host}:{args.port}/health")
    print("  First start takes ~30s (loading the cross-encoder). Ctrl+C to stop.\n")
    uvicorn.run("research_rag.api.app:app", host=args.host, port=args.port,
                reload=args.reload, workers=None if args.reload else args.workers)


if __name__ == "__main__":
    main()
