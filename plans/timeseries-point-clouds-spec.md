# Point clouds from time series: spec

Issue: [#267](https://github.com/kthtda/stablebear/issues/267)

Plan: [timeseries-functions-plan.md](timeseries-functions-plan.md), phase B

## Goal

Persistent homology of a time series works on point clouds sampled from it,
such as the points a delay embedding or its windows trace. Today users
sample by hand, evaluating every window at a grid of times. One call should
give a `PointCloud` or `PointCloudTensor` in a few standard modes.

## Decisions

These settle the open questions of the issue.

1. **Signature.** One function, `to_point_clouds`, whose arguments choose
   the mode: none for every observation, `n` or `step` for a uniform grid,
   `times` for given times.
2. **Scalar observations.** A series of numbers gives one-dimensional point
   clouds, one point of dimension 1 per time.
3. **Channels.** `sb.stack` combines the channels of a series into one, as
   it stacks tensors, and `sb.concatenate` as it joins them, so that
   channels of different lengths can share one vector. Both take one
   `TimeSeries`, whose channels they combine. A list of series raises
   `TypeError`, naming `TimeSeries.from_channels` and `sb.tensor`: for
   tensors, `sb.stack([x, y])` puts `x` and `y` side by side, as
   `sb.tensor([x, y])` does, so for series that form is left free to mean
   a tensor of series.
   (First built as `TimeSeries.stack_channels` and
   `TimeSeries.concatenate_channels`, then moved into `sb.stack` and
   `sb.concatenate` to avoid separate names for one operation.) `to_point_clouds`
   takes series of one channel only. One cloud per channel is
   `to_point_clouds(series[i])`.
4. **Ends of the domain.** "Every observation" takes the times a series
   stores that lie in its domain. A window stores the value in effect at its
   start at time 0; that point is included if the window includes its start,
   and left out if the start is open, though the value holds just after it.
   The end adds no point: the value there is that of the last observation,
   or is itself stored if the end is closed.

## Behavior

### `sb.stack(series, axis=0)`

- Returns a series of one channel whose observation at each time stacks
  the channels' observations in effect there along a new axis `axis` of
  length `n_channels`, as `sb.stack` stacks tensors: shape `(n_channels,
  *shape)` for `axis=0`. Any element type works. A negative `axis` counts
  from the end.
- The times are those at which any channel stores an observation within
  the domain, and the start of the domain. The domain stays the same, with
  its ends open or closed as before.
- Observations can be dropped or repeated, and the docstring warns of it:
  a channel's observations outside the domain, or at an end the domain
  leaves out, are dropped, and an observation in effect at several of the
  times is repeated at each.
- The unit and epoch stay the same. The snap tolerance is the smallest of
  the channels'.
- Raises `ValueError` if the channels have different shapes, as `sb.stack`
  does, or different origins, if `axis` is out of range, if their domains do
  not overlap, or if a channel has no observations.

### `sb.concatenate(series, axis=0)`

- As `sb.stack`, with the same times, domain, unit, epoch, origin,
  and snap tolerance, but joining the channels' observations along their
  existing axis `axis`, as `sb.concatenate` joins tensors: channels of
  shapes `(2,)` and `(3,)` give observations of shape `(5,)`. A negative
  `axis` counts from the end.
- Raises `ValueError` as `sb.stack` does, except that the shapes may
  differ along `axis`; also if the channels are scalars, naming
  `sb.stack`, or have different numbers of axes.

### `TimeSeries.restrict(domain)`

Tests of point clouds need series with open ends, which only `cut_windows`
made before, at shifted times.

- Returns the series on `domain`, a `Domain` within its own, whose ends are
  durations from time 0, or numbers for a series without a unit, with
  `start_closed` and `end_closed`.
- A series with an epoch raises `TypeError`, naming
  `series.with_epoch(None).restrict(domain).with_epoch(series.epoch)`, so
  that the ends are always in the series' own time and the same for every
  channel, whatever its origin.
- Times keep their place: the unit, epoch, and origin stay the same. Each
  channel holds the observation in effect at the new start, at the start,
  then its observations inside the domain, and one at the end if it is
  closed, as a window does. Observations are shared, not copied.
- Raises `ValueError` if `domain` reaches outside the series' domain,
  including an end the series leaves out but `domain` includes.
  Raises `TypeError` for ends of the wrong kind, as `evaluate` does.
- `Domain` itself raises `ValueError` if it holds no time: its start comes
  after its end, or equals it without both ends closed. It stays a named
  tuple, through a private base class whose constructor it checks.
- Not covered: extending a series past its last observation.

### `to_point_clouds(series, times=None, *, n=None, step=None)`

In `stablebear.timeseries`.

- `series` is a `TimeSeries`, a `TimeSeriesTensor` such as windows, or a
  pandas `Series` or `DataFrame`, converted with `TimeSeries.from_pandas`.
- Observations must be numbers of shape `()` or `(m,)`, giving points of
  dimension 1 or `m`. `float32` gives `pcloud32`; other numbers give
  `pcloud64`. Other element types or shapes raise `TypeError`.
- A `TimeSeries` gives a `PointCloud`, and a `TimeSeriesTensor` of shape
  `S` a `PointCloudTensor` of shape `S`, whose clouds can differ in size.
- A series of several channels raises `ValueError`, naming
  `sb.stack` and `series[i]`.
- The points come in the order of their times.

### Modes

- **Every observation** (no `times`, `n`, or `step`): one point per time the
  series stores in its domain.
- **Uniform grid, `n`**: `n` evenly spaced times over each series' domain,
  in its own time from 0, including each end the domain includes. With both
  ends closed they are `length / (n - 1)` apart, with one end open
  `length / n` apart starting from the closed end, and with both open
  `length / (n + 1)` apart. These are the evenly spaced times over the
  domain with both ends closed, with `n + 1` or `n + 2` times when ends are
  open and the open ends left out, so `[0, 2]` gives 0, 1, 2 and `[0, 2)`
  gives 0, 2/3, 4/3. The clouds of consecutive half-open windows then form
  one evenly spaced grid, without a time in two clouds. Windows of one size
  give clouds of `n` points each. `n` is a positive integer; `n = 1` needs an open end, since one
  point cannot include both ends; it lies at the closed end, or in the
  middle if both are open.
- **Uniform grid, `step`**: times `start + k * step` in each series'
  domain, for `k = 0, 1, ...`, leaving out `start` if that end is open.
  `step` is a positive duration, or a number for a series without a unit,
  as for `cut_windows`.
- **Given times** (`times`): the same times for every series, as
  `evaluate` takes them: durations, numbers for a series without a unit, or
  dates for a series with an epoch. A time outside a domain raises
  `ValueError`, as `evaluate` does.

### Errors

- `ValueError` if more than one of `times`, `n`, and `step` is given, for
  an invalid `n` or `step`, or for several channels.
- `TypeError` for unsupported observations, and for numbers on a series
  with a unit or durations on one without, as `evaluate` raises.

## Out of scope

- Random sampling: users subsample the resulting point clouds.
- Weights or times as an extra coordinate of the points.
