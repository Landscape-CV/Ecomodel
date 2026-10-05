"""FastAPI app: serves the web UI and the annotator API.

  cd SyntheticPipeline
  python -m instance_annotator            # opens http://127.0.0.1:8765
  python -m instance_annotator --load testdataset/real_instance/l1w_t00_03
  python -m instance_annotator --project D:/pointclouds/island_project
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

_SP_DIR = Path(__file__).resolve().parents[2]
_ROOT = _SP_DIR.parent
for p in (str(_ROOT), str(_SP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from instance_annotator.server import project_routes  # noqa: E402
from instance_annotator.server.routes import router, session  # noqa: E402

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def create_app() -> FastAPI:
    app = FastAPI(title="TLS Instance Annotator", docs_url="/api/docs")
    app.include_router(router)
    app.include_router(project_routes.router)

    @app.middleware("http")
    async def no_cache(request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("Cache-Control", "no-cache")
        return resp

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/", StaticFiles(directory=str(WEB_DIR)), name="web")
    return app


app = create_app()


def main(argv=None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="TLS instance annotator (web UI)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--load", default="", help="Tile prefix or cloud to open on start")
    ap.add_argument("--blank", action="store_true", help="Ignore *_instances.npy when loading")
    ap.add_argument("--max_display", type=int, default=None, help="Display LOD point cap")
    ap.add_argument("--no_browser", action="store_true")
    ap.add_argument("--project", default="", help="Island project folder to open on start")
    args = ap.parse_args(argv)

    if args.project:
        info = project_routes.psession.open(args.project)
        print(f"Project {info['name']}: {len(info['tiles'])} tiles, {info['num_points']:,} pts")
    if args.load:
        print(f"Loading {args.load} ...")
        info = session.load(args.load, "blank" if args.blank else "gt", args.max_display)
        print(f"  {info['num_points']:,} pts, display {info['num_display']:,}, trees={info['num_trees']}")

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Annotator at {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
