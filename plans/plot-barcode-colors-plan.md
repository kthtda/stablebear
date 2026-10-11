# plot_barcode colors (#215): review follow-up plan

This plan finishes PR #215 on `plot_barcode_color_fix`. The PR lets
`plot_barcode` take a sequence of colors or a `cmap` for barcode tensors,
cycles default colors per barcode, matches infinite-bar arrowheads to their
bar color, and rejects `colors`. Its 12 new tests pass on the branch after
merging main. The items below come from reviewing that branch.

## How to work through this plan

Work through the items in order: the decisions shape the code changes, and
the tests and docs describe the final behavior. Each item is one review unit.
Keep new work uncommitted for review, and check an item off only after we have
reviewed it. Commit on top of MattiasBjerklov's commits without rewriting
them: no rebase or force-push on this branch.

Read [TESTING.md](../test/TESTING.md) before changing tests and
[WRITING.md](../docs/WRITING.md) before changing documentation. Build and
test commands are in [AGENTS.md](../AGENTS.md#testing). `plotting.py` is pure
Python, so `PYTHONPATH=..` from `test/` runs the tests against the branch
source without reinstalling.

Each item gives the change, the expected behavior, and **Catches:**, the bug a
test would catch. Tags: **(decision)** a choice to make before the items that
follow, **(optional)** safe to skip.

## Decisions

- [x] **1. Where default colors come from.** (decision: (b), the axes' own cycle) With no `color` or
  `cmap`, a tensor currently takes colors from the global
  `rcParams["axes.prop_cycle"]`, starting at C0 on every call. So
  `ax.set_prop_cycle(...)` is ignored, and a second `plot_barcode` call on the
  same axes reuses the first call's colors. Options:
  (a) keep the global cycle and document that each call starts at C0;
  (b) draw from the axes' own cycle, as `ax.plot` does
  (`ax._get_lines.get_next_color()`), so `set_prop_cycle` applies and
  repeated calls continue the cycle. Recommendation: (b), which matches
  Matplotlib's conventions. It relies on a private Matplotlib attribute, so
  check that it exists in the oldest supported Matplotlib.
- [x] **2. Default color for a single barcode.** (decision: (b), the next color from the axes' cycle) A single `Barcode`
  or array with no color always gets C0, while tensor elements cycle. Options:
  (a) keep C0 and document it; (b) take the next color from the same source as
  item 1, so separate single-barcode calls on one axes get different colors.
  Recommendation: follow item 1. With (b) there, use (b) here, so a single
  barcode behaves like a 1-element tensor, as the docstring already claims.
- [x] **3. Three or four numbers as `color`.** (decision: keep Matplotlib's rule, one color) `color=(0.5, 0.5, 0.5)`
  is a single RGB color, never three per-barcode colors, because Matplotlib
  treats it as color-like. Recommendation: keep this and document it, since
  a per-barcode sequence can always be written as a list of named or hex
  colors.

## Code

- [x] **4. A clear error for non-color, non-sequence values.**
  `_resolve_barcode_colors` calls `list(color)` on anything that is not
  color-like, so `color=5` raises Python's
  "'int' object is not iterable". Raise the documented ValueError,
  "'color' must be a color or a non-empty sequence of colors", for any value
  that is neither color-like nor an iterable of color-like values.
  **Catches:** raw TypeErrors that contradict the docstring.
- [x] **5. Implement the defaults chosen in items 1 and 2.** Update
  `_resolve_barcode_colors` and the single-barcode path. If item 1 picks (b),
  the helper needs the axes, so resolve `ax` before resolving colors. That
  must still happen after input validation, so invalid calls do not create an
  empty figure (a guarantee from #159).
- [x] **6. Define `arrow_color` unconditionally.** (done: initialized to None before the check) It is assigned only inside
  `if segments:`. This is correct today, since infinite bars always add
  segments, but fragile. Compute it from the resolved color, or initialize it
  before the branch. No behavior change.
- [x] **7. Docstring cleanup.** Capitalize "Number of barcodes" in the
  `_resolve_barcode_colors` docstring. Update the `plot_barcode` docstring for
  the decisions in items 1-3.

## Tests

These go in `test/python/test_plotting.py`, class `TestPlotBarcode`.

- [x] **8. Split multi-behavior tests.** `test_colors_kwarg_rejected` covers a
  single barcode and a tensor; `test_tensor_invalid_color_raises` covers an
  invalid name and an empty list; `test_color_and_cmap_together_raises`
  covers a tensor and a single barcode. Parametrize each over its cases, so a
  failure names the case.
- [x] **9. Non-iterable colors raise ValueError.** Parametrize over `5` and
  `object()`; expect the item 4 message. **Catches:** the raw TypeError
  coming back.
- [x] **10. Default colors follow the decisions.** Replace or extend
  `test_tensor_default_colors_follow_prop_cycle` to match items 1 and 2. If
  (b): after `ax.set_prop_cycle(color=["m", "y"])`, a two-barcode tensor uses
  magenta, then yellow; a second call continues the cycle; and two
  single-barcode calls get different colors. **Catches:** the global cycle
  being used instead of the axes' cycle.
- [x] **11. An RGB tuple is one color.** A three-barcode tensor with
  `color=(0.5, 0.5, 0.5)` draws all three in that gray. **Catches:** an RGB
  tuple split into per-barcode values.
- [x] **12. Arrowheads use the default color.** (optional) An infinite bar
  with no `color` gets an arrowhead matching the bar's resolved default
  color; the existing arrowhead test covers only an explicit color with alpha.

## Documentation and release notes

- [x] **13. Document the color options.** In `docs/plotting.rst`, under the
  barcode section, describe `color` (single or per-barcode), the default
  cycling, how to pass a colormap's colors through `color`, and that `colors`
  and `cmap` are rejected. Add a short example with a barcode tensor colored
  from a colormap, for ordered barcodes such as a noise sweep. Following AGENTS.md, a new figure needs a
  `.. dropdown:: Show code` with a `literalinclude` of a snippet in
  `docs/_static/gen_plotting_gallery.py`.
- [ ] **14. Changelog.** Add 0.5.1 entries in `CHANGELOG.md`, linking #215:
  under New features, per-barcode `color` sequences and default color
  cycling; under Bug fixes, arrowheads matching their bar's color and alpha, and
  infinite bars in a barcode tensor ending at the same x.
  Rejecting `colors` and `cmap` changes existing calls, so list it under
  Breaking changes.

## Final checks

- [ ] **15. Test run.** From `test/`, run `test_plotting.py` and
  `test_plotting_heatmap.py` against the branch source, then the full Python
  suite after a CMake build and install. Regenerate the gallery with
  `python docs/_static/gen_plotting_gallery.py` and check that the barcode
  figures are unchanged, since they pass single colors.
- [ ] **16. Docs build.** Run `make html` in `docs/` (with
  `LC_ALL=C.UTF-8` if the shell locale is missing) and check for warnings.
- [ ] **17. Push.** Push to `plot_barcode_color_fix` as ordinary commits, then
  remove this plan before merging.
