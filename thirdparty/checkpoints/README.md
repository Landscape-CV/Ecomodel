# Shared model checkpoints

Place downloaded weights here (gitignored). Suggested layout:

```text
thirdparty/checkpoints/
  README.md
  point_sam/
    model.safetensors          # Point-SAM ViT-L from HuggingFace
  snap/
    SNAP_C.pth                 # SNAP Outdoor / C checkpoint
```

Machine-specific install + verify steps:

- [`../Point-SAM/INSTALL_ECOMODEL.md`](../Point-SAM/INSTALL_ECOMODEL.md)
- [`../SNAP/INSTALL_ECOMODEL.md`](../SNAP/INSTALL_ECOMODEL.md)
