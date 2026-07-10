"""
SmartQSM on Modal — serverless GPU backend for the packaged Ecomodel GUI.

Deploy once (the lab); the packaged GUI then points at the printed URL:

    pip install modal
    python3 -m modal setup                 # one-time browser auth
    modal deploy deploy/modal_app.py       # builds image, prints the web URL

Set that URL as SMARTQSM_URL (or the GUI's smartqsm_url config field). The Mac
runner (ecomodel/gui/smartqsm_runner.py) already speaks this endpoint's protocol
unchanged:

    POST /reconstruct   body = raw .npy of (N,3) float32 points -> {"job_id": ...}
    GET  /result/<id>   202 while running, 200 + raw .npy Cx8 when done, 500 on error

Why this shape:
  * The GPU work runs as a Modal function that scale-to-zero ($0 when idle; bills
    only GPU-seconds used). The web layer spawns it and polls its result, so no
    single HTTP request has to stay open for the whole ~4-min reconstruction.
  * The image bakes the EXACT proven stack (py3.11 + torch cu121 + spconv-cu121 +
    the SmartQSM repo + its bundled checkpoints), so there is no per-user setup
    and no arm64/spconv problem — that all lives in a Linux+CUDA container.
"""
import modal

APP_NAME = "smartqsm"
REPO = "/opt/SmartQSM"
CONFIG = "configs/spconv-contraction-LEAFON-GPU.yaml"   # the real deep-learning method
GPU_TYPE = "T4"                                          # cheapest that works; "L4"/"A10G" faster
SMARTQSM_GIT = "https://github.com/project-lightlin/SmartQSM.git"


def _patch_repo():
    """Apply the 3 py3.11 f-string fixes (SmartQSM uses py3.12-only syntax).

    Runs at image-build time (via .run_function). Mirrors the Colab notebook's
    patch cell; without it, smartqsm.py / _updater.py fail to compile on 3.11.
    """
    patches = {
        f"{REPO}/entrypoints/smartqsm.py": [
            ('print(f"Skipped files: \\n{"\\n".join(skipped_files)}")',
             'print("Skipped files: \\n" + "\\n".join(skipped_files))'),
            ('print(f"Failed files: \\n{"\\n".join(failed_files)}")',
             'print("Failed files: \\n" + "\\n".join(failed_files))'),
        ],
        f"{REPO}/entrypoints/_updater.py": [
            ('f"Unsupported media_type {desc.get("media_type")}"',
             'f"Unsupported media_type {desc.get(\'media_type\')}"'),
        ],
    }
    for path, reps in patches.items():
        s = open(path).read()
        for old, new in reps:
            if old in s:
                s = s.replace(old, new)
        open(path, "w").write(s)
    import py_compile
    for path in patches:
        py_compile.compile(path, doraise=True)


# ── Container image: the exact proven SmartQSM stack (built in Modal's cloud) ──
# torch + spconv CUDA versions MUST match (cu121 both) or spconv SIGFPEs on GPU.
# SmartQSM's requirements.txt does NOT pin torch/spconv, so installing it after
# our cu121 torch is safe (it won't clobber it).
#
# TODO(reproducibility): pin versions so a future rebuild is byte-stable — a fixed
# SmartQSM commit (clone a tag or checkout a SHA instead of --depth 1 latest), a
# pinned spconv-cu121 version, and pinned requirements. torch is pinned; the rest
# currently floats, so a months-later rebuild could pull different versions.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "xvfb", "python3-tk", "libgl1", "libglib2.0-0")
    .run_commands(
        "pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121",
        "pip install spconv-cu121 --only-binary=:all:",
    )
    .run_commands(f"git clone --depth 1 {SMARTQSM_GIT} {REPO}")
    .run_commands(f"pip install -r {REPO}/requirements.txt")
    .pip_install("scipy", "numpy", "fastapi[standard]")
    .run_function(_patch_repo)
)

app = modal.App(APP_NAME, image=image)


def _parse_qsm_mat(mat_path):
    """SmartQSM *_qsm.mat -> Cx8 [start(3), radius, axis(3), length]."""
    import numpy as np
    import scipy.io as sio
    m = sio.loadmat(mat_path, simplify_cells=True)
    qsm = m.get("QSM", m)
    cyl = qsm["cylinder"]
    n = int(np.asarray(cyl["radius"]).ravel().shape[0])
    start = np.asarray(cyl["start"], float).reshape(n, 3)
    axis = np.asarray(cyl["axis"], float).reshape(n, 3)
    radius = np.asarray(cyl["radius"], float).reshape(n, 1)
    length = np.asarray(cyl["length"], float).reshape(n, 1)
    return np.concatenate([start, radius, axis, length], axis=1)


# ── GPU function: one segment in, Cx8 out. Scales to zero when idle. ──────────
@app.function(gpu=GPU_TYPE, timeout=1800)
def run_smartqsm(points_bytes: bytes) -> bytes:
    import io
    import os
    import subprocess
    import uuid
    import numpy as np

    pts = np.load(io.BytesIO(points_bytes))
    job = f"/tmp/{uuid.uuid4().hex}"
    os.makedirs(job, exist_ok=True)
    xyz = os.path.join(job, "seg.xyz")
    np.savetxt(xyz, np.asarray(pts, float), fmt="%.6f")

    cmd = ["xvfb-run", "-a", "python", "entrypoints/smartqsm.py",
           "-t", "-y", "-c", CONFIG, xyz]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)

    mat = os.path.join(job, "seg_qsm.mat")
    if not os.path.exists(mat):
        raise RuntimeError("SmartQSM produced no .mat.\nSTDERR tail:\n" + r.stderr[-2000:])

    cx8 = _parse_qsm_mat(mat)
    buf = io.BytesIO()
    np.save(buf, cx8.astype("float64"))
    return buf.getvalue()


# ── Web layer: the endpoints the Mac runner already calls ────────────────────
@app.function()
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request, Response

    api = FastAPI()

    @api.get("/health")
    def health():
        return {"ok": True, "config": CONFIG, "gpu": GPU_TYPE}

    @api.post("/reconstruct")
    async def reconstruct(request: Request):
        body = await request.body()
        call = run_smartqsm.spawn(body)          # kick off the GPU job, don't wait
        return {"job_id": call.object_id}

    @api.get("/result/{job_id}")
    def result(job_id: str):
        fc = modal.FunctionCall.from_id(job_id)
        try:
            out = fc.get(timeout=0)              # non-blocking poll
        except TimeoutError:
            return Response(status_code=202)     # still running
        except Exception as e:                   # GPU job failed / result expired
            return Response(content=str(e)[:1500], status_code=500)
        return Response(content=out, media_type="application/octet-stream")

    return api


@app.local_entrypoint()
def main():
    print("Deploy with:  modal deploy deploy/modal_app.py")
    print("Then set SMARTQSM_URL to the printed web URL and use the GUI / test client.")
