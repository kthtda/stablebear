# Windows and delay embeddings as functions

## Goal

Windows and delay embeddings were made by classes whose instances held
their arguments and were applied to a series by calling them. Nothing used
an instance more than once, and the rest of stablebear exposes operations
on data as functions, such as `pdist` and `compute_persistent_homology`.
Both are made by functions that take the series first.

## Behavior

- `cut_windows(series, size, step=None, *, closed=None, partial=False,
  anchor="end")` returns the windows of `series`, as the class did.
- The windows' bounds are no longer returned with them. A new property,
  `TimeSeries.source_domain`,
  gives where a window lies in the series it was cut from: the domain of
  each channel moved by its origin, intersected, as a `Domain` of
  durations, numbers, or dates. A series built from observations has its
  domain as its source domain.
- `delay_embed(series, dimension, delay, *, direction="backward")` returns
  the delay embedding of `series`, as the class did.
- Both take a `TimeSeries` or a pandas `Series` or `DataFrame`, and the
  same durations and numbers as before. Invalid arguments raise the same
  errors, now when the function is called.
- The classes are removed. The time series has only appeared in the
  0.6.0 betas, so no deprecation is needed.

## Out of scope

- Any change to what windows and embeddings compute.
