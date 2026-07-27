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
    """Read SmartQSM's output file and flatten it into a plain number table.

    SmartQSM writes its answer as a MATLAB `.mat` file containing one entry per
    fitted cylinder. Each cylinder is described by four things: where it starts
    (a 3D point), how thick it is (radius), which way it points (a 3D direction),
    and how long it is.

    This pulls those four fields out and glues them side by side into a single
    C-by-8 table (C = number of cylinders), one row per cylinder:

        [start_x, start_y, start_z, radius, axis_x, axis_y, axis_z, length]

    That Cx8 layout is what the Ecomodel GUI expects, so this is the translation
    step between SmartQSM's format and ours.
    """
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
@app.function(gpu=GPU_TYPE, timeout=3600)
def run_smartqsm(points_bytes: bytes) -> bytes:
    """Reconstruct ONE tree on a GPU. Points in, cylinders out.

    This is the only function that touches the GPU, and it is the whole reason
    Modal is here: SmartQSM's neural network needs CUDA + Linux, which the Mac
    cannot provide.

    The `@app.function(gpu="T4", ...)` line above is what makes that happen. When
    this function is called, Modal boots a container with a T4 GPU, runs the body,
    then shuts it down. Nothing is running (or being billed) in between.

    What it does, step by step:
      1. Take the raw bytes of a .npy file and turn them back into an (N,3) array
         of XYZ points. This is one tree's worth of points, sent by the Mac.
      2. Write them to a scratch .xyz text file, because SmartQSM is a
         command-line tool that reads files, not a Python library we can call.
      3. Shell out to SmartQSM and wait for it to finish. `xvfb-run` fakes a
         display, because SmartQSM insists on loading GUI libraries at startup
         even when running headless.
      4. SmartQSM leaves its answer in a `seg_qsm.mat` file next to the input.
         If that file is missing, the run failed, so raise with its error output.
      5. Convert the .mat into the Cx8 table and hand it back as .npy bytes.

    Args:
        points_bytes: a .npy file, as raw bytes, holding an (N,3) float array.

    Returns:
        A .npy file, as raw bytes, holding the Cx8 cylinder table.

    Note the 3600s (60 min) timeout in the decorator. Modal kills the job at that
    point. One tree finishes in a few minutes; the extra headroom is for a single
    very large tree (e.g. a big mangrove) that can't be split further. A whole
    un-segmented multi-tree tile still won't finish — segment first. See
    Docs/KNOWN_ISSUES.md.
    """
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
    """Build the little web server the Mac talks to. Runs on CPU, not the GPU.

    This function is not the website. It *creates* the website (a FastAPI app)
    and returns it, and Modal then serves it at a permanent public URL. That URL
    is the one baked into the GUI's `smartqsm_url` config default, which is why
    users never have to configure anything.

    THE KEY IDEA — why there are two endpoints instead of one:

    A reconstruction takes minutes. If the Mac made one HTTP request and waited
    for the answer, the connection would time out long before the GPU finished.
    So the work is split in two, which is the standard "submit and poll" pattern:

        POST /reconstruct  ->  "here are the points, start working"
                               replies INSTANTLY with a job_id ticket
        GET  /result/<id>  ->  "is job <id> done yet?"
                               asked over and over, every few seconds

    Each individual request finishes in milliseconds. The long wait happens on
    Modal's side, not inside an open connection.
    """
    from fastapi import FastAPI, Request, Response

    api = FastAPI()

    @api.get("/health")
    def health():
        """Is the service alive? Returns which GPU and config it is set up for.

        Handy for `curl <url>/health` after deploying, to check the URL works
        before involving the GUI. Does not touch the GPU, so it costs nothing.
        """
        return {"ok": True, "config": CONFIG, "gpu": GPU_TYPE}

    @api.post("/reconstruct")
    async def reconstruct(request: Request):
        """Accept one tree's points and start a GPU job. Does NOT wait for it.

        `.spawn()` is the important bit: it launches run_smartqsm in the
        background and returns immediately, instead of blocking until it's done.
        We hand back the job's id so the caller can ask about it later.
        """
        body = await request.body()
        call = run_smartqsm.spawn(body)          # kick off the GPU job, don't wait
        return {"job_id": call.object_id}

    @api.get("/result/{job_id}")
    def result(job_id: str):
        """Check on a job. One of three answers, depending on how it's going.

        `fc.get(timeout=0)` means "give me the result, but don't wait even a
        moment for it". So:

            still running  -> it raises TimeoutError -> we reply 202 Accepted,
                              which tells the caller "not yet, ask again"
            finished       -> we reply 200 with the Cx8 cylinder table as bytes
            blew up        -> we reply 500 with the error text

        The Mac's runner keeps calling this every 4 seconds until it gets a 200.
        """
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
    """Just a reminder, printed if you run this file directly instead of deploying.

    `python deploy/modal_app.py` does nothing useful. The file is meant to be
    handed to Modal with `modal deploy`, which uploads it, builds the image in
    Modal's cloud, and prints the public URL.
    """
    print("Deploy with:  modal deploy deploy/modal_app.py")
    print("Then set SMARTQSM_URL to the printed web URL and use the GUI / test client.")
