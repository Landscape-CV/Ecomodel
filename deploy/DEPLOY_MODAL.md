# Deploy SmartQSM on Modal — runbook

## What this does

SmartQSM is the deep-learning method the Ecomodel GUI uses to reconstruct trees
into cylinder models (QSMs). This deploys SmartQSM as a **remote GPU service**
that the GUI calls over the internet. Deploy it **once** (the lab), and every
copy of the GUI reconstructs trees on that GPU with **no setup from the user** —
they just pick "SmartQSM" as the QSM method and hit Start.

```
[Mac GUI] --HTTP--> [Modal web endpoint] --spawn--> [Modal GPU function: real spconv NN]
   render cylinders  <--Cx8-- poll /result/<id> <----------------------------
```

## Why a remote serverless GPU (and not the laptop)

- **SmartQSM's core needs a GPU.** Its neural network uses a CUDA-only library
  (`spconv`) that cannot run on an Apple Silicon Mac at all, and a GPU can't be
  bundled inside a desktop app. So the GUI *must* send the work to a GPU
  elsewhere.
- **Usage is occasional, not constant.** Ecologists process a batch of trees now
  and then. A GPU server running 24/7 would sit idle almost all the time and
  cost money for nothing.
- **Serverless solves both.** The GPU spins up only when a reconstruction is
  requested and **scales back to zero when idle** — so it's always available at a
  stable URL, but costs **$0 when no one is using it** and only bills for the
  seconds of GPU actually spent.

## What is Modal

[Modal](https://modal.com) is a serverless-GPU platform. You describe a container
(here: Python 3.11 + torch + spconv + the SmartQSM code) and a function to run on
a GPU, then `modal deploy` builds it in Modal's cloud and gives you a permanent
HTTPS URL. When a request hits that URL, Modal pulls a GPU, runs your function,
and releases the GPU afterward. You pay per GPU-second used; idle is free (a free
tier covers light research use). The whole environment lives in Modal's Linux+CUDA
container, which is what sidesteps the "spconv won't run on the Mac" problem.

`deploy/modal_app.py` defines all of this: the container image, the GPU function
(`run_smartqsm`), and the web endpoints the GUI calls. `deploy/test_gpu_service.py`
is a GUI-free client for testing the endpoint directly.

## How it works

**One-time setup (the lab does this once):**

```
  modal deploy deploy/modal_app.py
        |
        v
  Modal builds the container image in its cloud
  (Python 3.11 + torch + spconv-cu121 + SmartQSM + checkpoints)
        |
        v
  Service is live at a stable URL:
  https://<workspace>--smartqsm-web.modal.run
        |
        v
  URL is baked into gui/config.py (smartqsm_url) -> the app ships ready to use
```

**Every reconstruction (each time a user runs SmartQSM in the GUI):**

```
  [Mac] GUI removes ground/leaves + (optionally) segments the tile
        |
        |  POST /reconstruct   (a tree's points, as .npy)
        v
  [Modal] web endpoint  --spawn-->  GPU job (run_smartqsm)
        |                                |  Modal pulls a GPU, loads the image
        |  GET /result/<job_id>          v
        |  (poll every few seconds)   real spconv neural net reconstructs -> cylinders
        v                                |
  cylinders (Cx8) returned  <------------+
        |
        v
  [Mac] GUI renders the QSM in the Results view
                                   (GPU scales back to zero when idle -> $0)
```

Why submit + poll (two endpoints) instead of one call: a reconstruction takes a
few minutes, longer than a single HTTP request should stay open. So `/reconstruct`
starts the job and returns a `job_id` immediately, and the client polls
`/result/<job_id>` until the cylinders are ready. From the GUI's side it still
feels like one synchronous call.

## Prerequisites (done)
- Modal account (workspace `main`).
- `modal` installed in the `base` conda env (`python -c "import modal"` works).
- App written + validated: `deploy/modal_app.py` (loads cleanly).

## Step 1 — Authenticate (one-time, needs your browser)
```bash
# base env is active by default; prompt shows (base)
python3 -m modal setup
```
Opens a browser to link the CLI to your account. Close the tab when done.

## Step 2 — Deploy
```bash
cd ~/Documents/Projects/lidar_project/ecomodel
python3 -m modal deploy deploy/modal_app.py
```
First run builds the image in Modal's cloud (installs torch cu121 + spconv-cu121,
clones SmartQSM, applies the py3.11 patches) — a few minutes. Re-deploys only
rebuild changed layers. When it finishes it prints a **web URL** for the `web`
function, e.g. `https://<workspace>--smartqsm-web.modal.run`. The URL is stable
across redeploys; it's also on the Modal dashboard under Apps -> smartqsm.

## Step 3 — Health check
```bash
curl https://<workspace>--smartqsm-web.modal.run/health
# -> {"ok":true,"config":"configs/spconv-contraction-LEAFON-GPU.yaml","gpu":"T4"}
```

## Step 4 — Prove the GPU path (reuse the test client)
```bash
conda activate pytlidar
python deploy/test_gpu_service.py ~/Downloads/segmented_trees/tree_43.las \
  --url https://<workspace>--smartqsm-web.modal.run --max-points 120000
```
Expect: submitted job -> polling -> DONE, a cylinder count, `% in 1-5 cm`, and a
`*_reconstruction.png`. First call is slower (cold start builds/loads the GPU
container); later calls while warm are fast.

## Step 5 — Point the GUI at it
The Modal URL is the default in `gui/config.py` (`smartqsm_url`), so the GUI routes
to Modal with **no setup**:
```bash
conda activate pytlidar
python run_gui.py
```
Lite pipeline -> QSM Method = SmartQSM -> Run. Leave the SmartQSM dir/python/config
fields blank. Same runner, same protocol as the Colab path. To point at a *different*
endpoint for testing, set `export SMARTQSM_URL=...` — it overrides the config default.
For a single tree, also tick **"Single tree (skip segmentation)"** so it reconstructs
in one piece instead of over-segmenting.

## Iterating
Edit `deploy/modal_app.py` -> `python3 -m modal deploy deploy/modal_app.py` again.
The URL stays the same. Use `python3 -m modal app logs smartqsm` to see build /
runtime logs.

## Likely first-deploy iteration points (normal)
| Symptom | Likely fix |
|---|---|
| Image build fails installing a requirement (e.g. `pywebview`) | Add the needed apt lib to `.apt_install(...)` per the build error (e.g. GTK/webkit libs), or drop the unused dep. |
| `run_smartqsm` errors: no `.mat` produced, spconv `Floating point exception` | torch/spconv CUDA mismatch. Both must be cu121 (they are); if it persists, pin a torch version known-good with the installed `spconv-cu121`. |
| `.xyz` read error in the SmartQSM log | The segment reader choked; try a different single-tree tile, or write `.las` instead of `.xyz` in `run_smartqsm`. |
| `/result` never returns 200 / raises on poll | Modal's `FunctionCall.get(timeout=0)` exception type differs by version; adjust the `except TimeoutError` branch in `web()` to the class Modal raises for "not ready". |
| Cold starts feel slow | Expected on first call; optionally set `scaledown_window` / keep-warm on `@app.function` to hold a warm GPU during a batch. |

## Cost
Image builds are cheap; each reconstruction is ~minutes of GPU = cents; **$0 when
idle** (scale-to-zero). Verify your plan includes the monthly free compute so
test runs stay within it.

## When it works -> package
Bake the URL into the GUI's config default (`ecomodel/gui/config.py` -> set
`smartqsm_url` default to the Modal URL) so the shipped app needs no env var.
Ship with TreeQSM (local, CPU) as the always-works fallback; SmartQSM activates
when the URL is reachable.

## Files
- `deploy/modal_app.py` — the Modal app (image + GPU function + web endpoints).
- Runner remote path — `ecomodel/gui/smartqsm_runner.py` (`_reconstruct_remote`).
- Test client — `colab_setup/test_gpu_service.py` (works against any URL).
