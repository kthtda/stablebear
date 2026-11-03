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
  anchor="end")` returns the windows that
  `Window(size, step, closed, partial, anchor)(series)` returned.
- `return_bounds` is dropped. A new property, `TimeSeries.source_domain`,
  gives where a window lies in the series it was cut from: the domain of
  each channel moved by its origin, intersected, as a `Domain` of
  durations, numbers, or dates. A series built from observations has its
  domain as its source domain.
- `delay_embed(series, dimension, delay, *, direction="backward")` returns
  what `TimeDelayEmbedding(dimension, delay, direction)(series)` returned.
- Both take a `TimeSeries` or a pandas `Series` or `DataFrame`, and the
  same durations and numbers as before. Invalid arguments raise the same
  errors, now when the function is called.
- `Window` and `TimeDelayEmbedding` are removed. The time series has only
  appeared in the 0.6.0 betas, so no deprecation is needed.

## Out of scope

- Any change to what windows and embeddings compute.
