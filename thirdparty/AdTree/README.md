# AdTree (Windows prebuilt)

Local install of [AdTree v1.1.2](https://github.com/tudelft3d/AdTree/releases/tag/v1.1.2) for the QSM benchmark. **Binaries are gitignored** — only this README is tracked.

## Install

1. Download `AdTree-v1.1.2_for_Windows.zip` from the release page above.
2. Extract into this folder so the layout is:

```text
thirdparty/AdTree/
  README.md
  AdTree-v1.1.2_for_Windows/
    AdTree.exe
    opengl32.dll
    LICENSE
    resources/
    ...
```

PowerShell one-liner from the repo root:

```powershell
$dest = "thirdparty/AdTree"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$zip = Join-Path $dest "AdTree-v1.1.2_for_Windows.zip"
curl.exe -L -o $zip "https://github.com/tudelft3d/AdTree/releases/download/v1.1.2/AdTree-v1.1.2_for_Windows.zip"
Expand-Archive -Path $zip -DestinationPath $dest -Force
```

3. Optional: install `vc_redist.x64.exe` from the zip if Windows reports a missing VCRUNTIME DLL.

Default executable path used by the benchmark:

`thirdparty/AdTree/AdTree-v1.1.2_for_Windows/AdTree.exe`

Override with `--adtree-exe` if needed.

## How the runner uses it

[`gui/adtree_runner.py`](../../gui/adtree_runner.py):

1. Writes the condition cloud as ASCII `tree.xyz` (normalized coords).
2. Runs: `AdTree.exe tree.xyz <work_dir> -s`
3. Treats **presence of `tree_skeleton.ply`** as success (AdTree **inverts exit codes**: success → 1).
4. Parses PLY vertices (`x,y,z,radius`) + edges → Cx9 cylinders (mean endpoint radius; BFS branch order from lowest-Z root).

License: GPL-3.0 (see `LICENSE` inside the release folder).
