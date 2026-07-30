# Known Issues

Bugs found while testing the Lite pipeline. Recorded here so they're visible to
whoever picks them up; none are fixed yet unless stated.

---

## 1. Stop button has no effect during the QSM step

**Severity:** high (user has to force-quit the app)

**Repro**
1. Lite pipeline, a large multi-tree tile (~5M points).
2. Tick *Single tree (skip segmentation)*, QSM method = TreeQSM.
3. Start. Once the log reaches `[6/6] QSM (treeqsm)...`, press **Stop**.
4. Nothing happens. Start and Stop both grey out and the app sits at
   "Step 6/7" indefinitely. The worker thread keeps burning CPU (observed at
   133% for many minutes). The only way out is to kill the process.

**Root cause**

`run_ecomodel_lite_pipeline` polls the stop flag only *between* steps —
`_check_stop()` is called at `gui/pipeline_lite.py:185, 195, 210, 225, 242, 259`,
i.e. before each of steps 1–5. Step 6 (the QSM fit) is a single long blocking
call into TreeQSM/SmartQSM with no callback, so once execution enters it the
flag is never read again. `should_stop` is cooperative; nothing can preempt it.

The QSM step is by far the longest step, so in practice Stop is unavailable
exactly when a user most wants it.

**Suggested fix**

Pass a cancellation check down into the QSM step so it can bail between trees —
`get_cylinders()` loops over instances (`ecomodel_lite.py`), so a per-instance
`should_stop()` check would make Stop responsive at tree granularity. Same for
`run_smartqsm_on_segments()`, which loops over segments POSTing to Modal.

Not a full fix (a single huge "tree" is still one long call), but it covers the
realistic case. A true fix needs the QSM work in a killable subprocess.

---

## 2. Multi-tile runs with mixed coordinate systems produce an unviewable point cloud

**Severity:** medium (silent — looks like a broken viewer, not a data problem)

**Repro**
1. Put two tiles in one input folder whose LAS files are in *different*
   coordinate frames — e.g. a UTM tile (`segtest_retile_...`) and a clipped
   single tree that was re-origined near zero (`tree_50.las`).
2. Run the Lite pipeline over both.
3. Results page → Point Cloud shows a single dot in an empty viewport.

**Root cause**

The queryable snapshot in `gui/pipeline_lite.py` concatenates every tile in world
coordinates and centres them all on ONE shared mean:

```python
world = np.concatenate([b[0] for b in snap_blocks], axis=0)
cloud_mean = world.mean(axis=0)
norm = (world - cloud_mean).astype(np.float32)
```

That is only correct if all tiles share a CRS. With mixed frames the mean lands
between the two clusters and the saved cloud spans an absurd extent — measured
~600 km in X on a real run, against a correct 35 m in Z. The viewer then fits the
camera to 600 km and the trees become sub-pixel. The data is fine; only the
framing is wrong.

**Suggested fix**

Cheapest: after building the snapshot, check the XY extent against something
plausible (say a few km) and log a loud warning naming the offending tiles.
Better: store a per-tile offset instead of one global mean, or reject a run whose
tiles' bounding boxes don't overlap.

**Workaround:** run one CRS at a time.

---

## 3. Scanline segmenter over-segments a single tree

**Severity:** high (blocks per-tree QSM — every downstream cylinder is fitted to
a fragment)

**Observed:** `tree_50.las`, leaf removal OFF → **20** segments; leaf removal ON →
**17**. The top-down view shows a single continuous canopy carved into contiguous
angular wedges that all meet in the middle — the signature of one crown being
split among competing bases, not of distinct trees. (Note tree_50 is itself a
multi-stemmed thicket rather than a clean single tree, so the raw count is not
proof on its own; the wedge pattern is.)

**Root cause candidates** (all in `Utils/TreeSegmentation.py`, `segment_point_cloud`)

Segment count is decided entirely in the base slab (`min_height`..`base_height`
above the lowest point): every cluster found there becomes a tree.

- `combine_nearby_bases` is forced to **False** by the `tuned_arguments` dict in
  `ecomodel_segmenters.py` (default is True). That disables `combine_close_bases`,
  which the source comment says exists precisely for "trees that have non-trunk
  segments that dip into base layer". It ALSO silently skips the stricter base
  filter, which only runs inside that branch.
- Base acceptance threshold is `len(base_set[:,2]) > 1` (`TreeSegmentation.py:201`)
  — a blob of **two cover sets** is promoted to a tree.
- `min_height` is documented as "minimum height of found segments" but is only
  used as the base-layer floor (lines 151, 179). **There is no vertical-extent
  filter on output segments anywhere**, so a ground-level blob with nothing above
  it is emitted as a tree.

**Structural concern:** the algorithm assumes one trunk = one blob near the
ground. Mangroves have prop roots and buttonwood is multi-stemmed, so the target
species genuinely present many "bases". The `treelearn` segmenter
(`ecomodel_segmenters.py:1052`) doesn't rely on this assumption and is worth
evaluating.

---

## 4. RGI leaf removal deletes fine branch structure

**Severity:** high for this project (the 1–5 cm band is the deliverable)

On `tree_50.las`, `remove_leaves_rgi` retained **9.7%** of points (498k → 61k).
Comparing before/after side views, what it removes is disproportionately the fine
peripheral crown branches — exactly the 1–5 cm structure SmartQSM is meant to
recover.

**Current mitigation:** `run_leaf_removal` defaults to False (`gui/config.py`).
Decision pending on whether that is the shipped default.
