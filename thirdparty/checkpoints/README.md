# Shared model checkpoints

Place downloaded weights here (gitignored). Suggested layout:

```text
thirdparty/checkpoints/
  README.md
  point_sam/
    model.safetensors          # Point-SAM ViT-L from HuggingFace
  snap/
    SNAP_C.pth                 # SNAP Outdoor / C checkpoint
  forestformer3d/
    epoch_3000_fix.pth         # clean_forestformer.zip from Zenodo 16742708
```

Machine-specific install + verify steps:

- [`../Point-SAM/INSTALL_ECOMODEL.md`](../Point-SAM/INSTALL_ECOMODEL.md)
- [`../SNAP/INSTALL_ECOMODEL.md`](../SNAP/INSTALL_ECOMODEL.md)
