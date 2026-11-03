# Windows and delay embeddings as functions

## Goal

`Window` and `TimeDelayEmbedding` are classes whose instances hold their
arguments and are applied to a series by calling them, as in
`Window(size=TimeDelta(2, "h"), anchor="start")(series)`. Nothing uses an
instance more than once, and the rest of stablebear exposes operations on
data as functions, such as `pdist` and `compute_persistent_homology`.
Both become functions that take the series first.

## Behavior

- `cut_windows(series, size, step=None, *, closed=None, partial=False,
  anchor="end", return_bounds=False)` returns what
  `Window(size, step, closed, partial, anchor)(series, return_bounds)`
  returned: the windows, and with `return_bounds=True` also their bounds.
- `delay_embed(series, dimension, delay, *, direction="backward")` returns
  what `TimeDelayEmbedding(dimension, delay, direction)(series)` returned.
- Both take a `TimeSeries` or a pandas `Series` or `DataFrame`, and the
  same durations and numbers as before. Invalid arguments raise the same
  errors, now when the function is called.
- `Window` and `TimeDelayEmbedding` are removed. The time series has only
  appeared in the 0.6.0 betas, so no deprecation is needed.

## Out of scope

- Any change to what windows and embeddings compute.
