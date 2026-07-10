"""
Standalone test client for the SmartQSM GPU service — the GUI-free fallback.

Sends a point cloud straight to the live Colab GPU service, gets cylinders back,
prints stats, and renders a points-vs-cylinders PNG. Use it to (a) prove the
tunnel + GPU path works before touching the GUI, and (b) have a presentable
live-GPU artifact even if the full GUI pipeline hiccups during the call.

Run in the `pytlidar` conda env (needs laspy + matplotlib):

    export SMARTQSM_URL="https://<something>.trycloudflare.com"
    python colab_setup/test_gpu_service.py /path/to/tree_43.las --max-points 120000

Or pass the URL explicitly with --url. Writes reconstruction.png next to the input.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from urllib import request as urlrequest

import numpy as np


def load_points(path, max_points=None):
    import laspy
    las = laspy.read(path)
    pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    pts = pts - pts.mean(axis=0)          # mirror the GUI's mean-subtraction
    if max_points and len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts


def reconstruct_remote(pts, url, timeout=1200, poll=4.0):
    base = url.rstrip("/")
    buf = io.BytesIO(); np.save(buf, pts.astype(np.float32))
    req = urlrequest.Request(base + "/reconstruct", data=buf.getvalue(),
                             headers={"Content-Type": "application/octet-stream"},
                             method="POST")
    with urlrequest.urlopen(req, timeout=120) as resp:
        job_id = json.loads(resp.read().decode())["job_id"]
    print(f"  submitted job {job_id}; polling for result...", flush=True)
    t0 = time.time()
    while True:
        with urlrequest.urlopen(base + "/result/" + job_id, timeout=120) as r:
            code = r.getcode(); data = r.read()
        if code == 200:
            return np.atleast_2d(np.load(io.BytesIO(data)))
        if time.time() - t0 > timeout:
            raise TimeoutError(f"job {job_id} exceeded {timeout}s")
        print(f"    ...running ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(poll)


def render(pts, cx8, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d  # noqa: F401

    starts = cx8[:, 0:3]
    radius = cx8[:, 3]
    axis = cx8[:, 4:7]
    length = cx8[:, 7]
    ends = starts + axis * length[:, None]

    fig = plt.figure(figsize=(11, 6))
    for k, (title, show_cyl) in enumerate([("point cloud", False),
                                           ("SmartQSM cylinders (GPU)", True)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        s = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 40000), replace=False)]
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=0.4, c="0.7", alpha=0.35, linewidths=0)
        if show_cyl:
            segs = np.stack([starts, ends], axis=1)
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            dia_cm = radius * 200.0
            lc = Line3DCollection(segs, linewidths=np.clip(radius * 300, 0.4, 4.0),
                                  array=dia_cm, cmap="viridis")
            ax.add_collection3d(lc)
        ax.set_title(title); ax.set_axis_off()
        try:
            ax.set_box_aspect((1, 1, 1.6))
        except Exception:
            pass
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud", help="path to a .las/.laz single-tree cloud")
    ap.add_argument("--url", default=os.environ.get("SMARTQSM_URL", ""))
    ap.add_argument("--max-points", type=int, default=120000)
    args = ap.parse_args()
    if not args.url:
        sys.exit("No service URL. Set SMARTQSM_URL or pass --url.")

    print("health:", end=" ", flush=True)
    with urlrequest.urlopen(args.url.rstrip("/") + "/health", timeout=30) as r:
        print(r.read().decode())

    pts = load_points(args.cloud, args.max_points)
    print(f"sending {len(pts)} points to {args.url} ...", flush=True)
    t0 = time.time()
    cx8 = reconstruct_remote(pts, args.url)
    dt = time.time() - t0

    dia_cm = cx8[:, 3] * 200.0
    in_band = np.mean((dia_cm >= 1.0) & (dia_cm <= 5.0)) * 100.0
    print(f"\nDONE in {dt:.0f}s")
    print(f"  cylinders     : {len(cx8)}")
    print(f"  diameter cm   : min {dia_cm.min():.2f}  median {np.median(dia_cm):.2f}  max {dia_cm.max():.2f}")
    print(f"  in 1-5 cm band: {in_band:.1f}%")

    out_png = os.path.splitext(args.cloud)[0] + "_reconstruction.png"
    render(pts, cx8, out_png)


if __name__ == "__main__":
    main()
