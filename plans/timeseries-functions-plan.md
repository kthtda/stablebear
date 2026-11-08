# Time series follow-ups: implementation plan

Progress: 18 of 20 items, 32 of 38 points (84%)

Specs: [timeseries-functions-spec.md](timeseries-functions-spec.md)
(phase A, #265) and
[timeseries-point-clouds-spec.md](timeseries-point-clouds-spec.md)
(phase B, #267). Phase C finishes #263, #264, and #266, which the
[pandas plan](pandas-timeseries-plan.md) implemented. Phase D builds
tensors of series, which `sb.stack` of series (B2c) is easily mistaken for.

## Phase A: functions instead of classes (#265)

- [x] **A1.** [3] (8d4f39e35, 84dea1a58) `cut_windows` replaces `Window` in
  `stablebear/timeseries/window.py`, with its docstring, and in the
  tests, internal callers, golden generator, and figure scripts, without
  `return_bounds`; `TimeSeries.source_domain` gives where a window lies.
  Tests: the existing window tests call `cut_windows`, and those of bounds
  check each window's `source_domain`; `source_domain` of a series built
  from observations, a dated window, a series without a unit, and
  channels at different origins; the tests of `Window.size`,
  `Window.step`, and its `repr` become tests that durations of each type
  give the same windows.
- [x] **A2.** [3] (664d1cefd) `delay_embed` replaces `TimeDelayEmbedding` in
  `stablebear/timeseries/embedding.py` likewise.
  Tests: as for A1, with `delay`.
- [x] **A3.** [2] (56c882342) The guide pages and the tide gauge tutorial use
  `cut_windows` and `delay_embed`, and show where windows lie by their
  `source_domain` instead of `return_bounds`.

## Phase B: point clouds from time series (#267)

- [x] **B1.** [1] (fb46ea98b) Write `plans/timeseries-point-clouds-spec.md`, settling
  the open questions of #267: the name and signature, what a series of
  scalar observations gives, how channels combine, and whether "every
  observation" includes the start of the domain and the end of a window.
- [x] **B2.** Combine the channels of a series into one.
  - [x] **B2a.** [2] (212620df5) `TimeSeries.stack_channels(axis=0)` stacks the channels of
    a series into one, as the
    [point cloud spec](timeseries-point-clouds-spec.md) describes. The work
    is in C++, in `include/sbear/timeseries/stack_channels.hpp`, using the
    C++ `stack` so that its shape check and `axis` match `sb.stack`; the
    Python method sits next to `from_channels`. Its docstring has a
    `.. warning::` that observations can be dropped or repeated: those
    outside the domain, or at an end the domain leaves out, are dropped, and
    one in effect at several of the times is repeated at each.
    Tests: scalar channels give vectors at the times of any channel in the
    domain and its start; vector channels with `axis` 0, 1, and -1; channels
    of PCFs; a window keeps its open start and origin; the unit and epoch
    stay; channels of different shapes or origins, and an `axis` out of
    range, raise.
  - [x] **B2b.** [2] (38231c712) `TimeSeries.concatenate_channels(axis=0)` joins the
    channels along an existing axis, as `sb.concatenate` does, sharing the
    C++ of `stack_channels` in the same header.
    Tests: it equals `sb.concatenate` of the channels' values at each time
    for floats and PCFs; channels of different lengths along `axis`; a
    negative `axis`; scalar channels raise naming `stack_channels`; shapes
    that differ off `axis`, and an `axis` out of range, raise.
  - [x] **B2c.** [2] (61313e35a) Move B2a and B2b into `sb.stack` and `sb.concatenate`,
    which take one `TimeSeries` (a list of series raises `TypeError`,
    naming `from_channels` and `sb.tensor`), and remove
    `TimeSeries.stack_channels` and `TimeSeries.concatenate_channels`. The
    C++ stays; errors name `sb.stack` and `sb.concatenate`. Overloads type
    a `TimeSeries` argument as giving a `TimeSeries`.
    Docs: a section in `timeseries_channels.rst`.
    Tests: the B2a/B2b tests, called through `sb.stack` and
    `sb.concatenate`; a list of series raises.
- [x] **B3.** [3] (c5b0afc5b) `to_point_clouds(series)` in its "every observation"
  mode: a series of one channel gives a `PointCloud` of one point per
  stored time in its domain, and a `TimeSeriesTensor`, such as windows, a
  `PointCloudTensor` whose clouds can differ in size. The work is in C++,
  in `include/sbear/timeseries/point_clouds.hpp`, with the tensor version
  filling the clouds with `parallel_walk`; name its parameter
  `seriesTensor`, not `series`. The bindings go in
  `register_derived_element`, behind a check that the element type is a
  number other than `bool`; the Python function only checks its argument
  and calls C++.
  Tests: a delay embedding gives its points; windows give clouds of their
  own sizes, including the value at time 0 only if the start is closed; partial windows give smaller
  clouds; scalars give 1-D clouds; integers give float64 clouds; a pandas
  `Series`; a series or windows of several channels raise, naming
  `stack_channels`; `float32` gives `pcloud32`; booleans and matrices
  raise.
- [x] **B4.** [2] (c3a2f1e7d) The uniform grid mode: `n` points, or points every
  `step`, over each series' domain from its time 0.
  Tests: `n` and `step` give the same cloud where they agree; `n` with
  closed, half-open, and open domains; windows of one size give clouds of
  one size; a step that does not divide the domain; an open start leaves
  out the start; more than one mode raises.
- [x] **B5.** [1] (247d9e6f0) The given-times mode: durations, or numbers for a series
  without a unit, as `evaluate` takes them.
  Tests: equals evaluating at those times; times outside the domain
  raise.
- [x] **B6.** [2] (5f2c54f56) A guide section on point clouds from time series, and
  the tide gauge tutorial builds its window clouds with it instead of
  evaluating on a grid by hand.
- [x] **B7.** [2] (e0ca36884) `TimeSeries.restrict(domain)`, as the
  [point cloud spec](timeseries-point-clouds-spec.md) describes; done
  before B4, whose tests build series with open ends with it. The C++
  cuts each channel with the code `windows` uses, moved into a function
  in `include/sbear/timeseries/windowing.hpp` that takes the time to
  subtract: the window start for `windows`, 0 for `restrict`. Python
  checks that `domain` lies within the series' own; a series with an
  epoch raises, pointing to `with_epoch(None)`. `Domain` raises if it holds
  no time.
  Tests: a closed series restricted to each kind of ends; a start between
  observations holds the value in effect there; the unit and origin
  stay; each channel is restricted as a series of its own; a domain
  reaching outside, ends of the wrong kind, and an epoch raise; a
  `Domain` of one time with both ends closed is valid, and one reversed,
  with an open end at one time, or with one time in two units raises.

## Phase C: changelog and documentation upkeep (#263, #264, #266)

- [x] **C1.** [1] (49ad56885) `CHANGELOG.md` cites #263 and #264 in the time series
  entries, and describes `cut_windows` and `delay_embed` (#265) and point
  clouds from time series (#267).
- [x] **C2.** Turn on Sphinx's nitpicky mode, so that a broken link to the
  API reference (#266) fails the docs build. A trial build with `-n` gives
  405 warnings, all unresolved classes in docstring type fields.
  - [x] **C2a.** [1] (1d74d648e) In `docs/conf.py`: `sphinx.ext.intersphinx` for
    Python, NumPy, pandas, and matplotlib; `napoleon_preprocess_types`,
    which drops words such as "optional"; `napoleon_type_aliases` mapping
    the short names of stablebear classes, such as `Tensor`,
    `TimeSeries`, and `FloatTensor`, to their documented paths; and
    `nitpick_ignore_regex` for private names, such as `_PcfNested`,
    `numpy._typing` internals, and the `_Mock` of mocked imports. The
    first two alone leave 301 warnings. Intersphinx fetches its
    inventories when the docs build, so the build needs the network.
    Tests: a build with `-n` warns only about the prose of C2b.
    Note: the type aliases `PcfContainerLike` and `ShapeLike`, the
    undocumented base `Tensor`, and `torch.Generator` are ignored too. The
    39 warnings left include unions and generics, such as
    `int | numpy.ndarray | DistanceMatrix` and `list[Rectangle]`, that
    napoleon does not split; C2b rewrites them with the prose.
  - [x] **C2b.** [2] (8e467e11a) Rewrite the type fields that hold prose, such as
    "durations", "them", "C++ tensor", "matplotlib axes", "file-like", and
    "same kind as ``X``", as types, with the prose moved into the
    description; document the dtype sentinels, such as `float32`, so that
    unions like `float32 | float64 | None` resolve, or ignore them.
    Tests: a build with `-n` gives no warnings.
    Note: the dtype sentinels resolve through napoleon aliases to
    `:data:` targets, since they are constants, not classes.
  - [x] **C2c.** [1] (d19e28959) `nitpicky = True` in `docs/conf.py` and
    `fail_on_warning: true` in `.readthedocs.yaml`.
    Tests: a broken reference fails a local build with `-W`.

## Phase D: tensors of time series

Today a `TimeSeriesTensor` comes only from `cut_windows`, `delay_embed`, or
loading a file: `sb.tensor([a, b])` fails with a NumPy error about too many
dimensions, and `sb.zeros` rejects `sb.ts(...)` dtypes. The C++ tensor
classes already take a shape, with an optional series to fill it, and
`TimeSeriesTensor.__setitem__` already takes a `TimeSeries`.

- [x] **D1.** [2] (2bb836cf9) `sb.zeros(shape, dtype=sb.ts(value_type))` gives a
  `TimeSeriesTensor` of empty series, through `_dtype_dispatch` in
  `stablebear/tensor_create.py`. An element built from the bare shape is
  a series of one channel with no observations and unit `"s"`; decide
  whether it should have no unit instead, and say what it is in the
  docstring.
  Tests: shape and dtype for float and PCF series; elements can be
  assigned a `TimeSeries`; an element left empty raises on evaluation.
- [ ] **D2.** [2] `sb.tensor` takes a list, or nested lists, of
  `TimeSeries` and gives a `TimeSeriesTensor` of that shape, as it does
  for lists of `Pcf`. Series may differ in times, lengths, units, and
  channels, but not in element type, and an explicit
  `dtype=sb.ts(value_type)` must match it.
  Tests: a list of two series of different lengths keeps each; nested
  lists give a 2-D tensor; series of different element types, or a
  `dtype` that does not match, raise `TypeError`.
- [ ] **D3.** [1] Docs: a "Tensors of series" section in
  `timeseries_channels.rst`, contrasting `sb.tensor([a, b])`, a tensor
  of two series, with `sb.stack(TimeSeries.from_channels([a, b]))`, one
  series of stacked observations; a note in `tensors.rst`, under "Joining tensors" and
  where `zeros` and `tensor` build tensors, that series go into a
  `TimeSeriesTensor` with `sb.tensor` or `sb.zeros`, while `sb.stack` and
  `sb.concatenate` combine their channels; a pointer from the `stack` and
  `concatenate` docstrings; a `CHANGELOG.md` entry.
- [ ] **D4.** [3] Dates and times for a tensor of series that all have an
  epoch, in `TimeSeriesTensor.evaluate` and `to_point_clouds`: each series
  places a date at `date - epoch - origin` in its own time, so separate
  series, such as one per station, are sampled at the same instants. C++
  takes the dates as integer counts of a unit and converts them per
  series. Today both raise `TypeError` for dates on a tensor; windows of
  one series rarely share a date in their domains, so they keep taking
  durations.
  Tests: two series with different epochs give the values at the same
  instants; a series without an epoch raises `ValueError`; a date outside
  a series' domain raises `ValueError` naming its position.
