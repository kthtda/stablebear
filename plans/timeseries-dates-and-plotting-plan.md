# Time series dates and plotting: implementation plan

Progress: 10 of 47 items, 21 of 89 points (24%)

Spec: [timeseries-dates-and-plotting-spec.md](timeseries-dates-and-plotting-spec.md)

Branch: `timeseries-improvements`. Follow the
[planning manifesto](PLANNING.md).

Python tests go in `test/python/test_timeseries.py`,
`test/python/test_windowing.py`, `test/python/test_time_delay_embedding.py`,
`test/python/test_io_roundtrip.py` and `test/python/test_pickle.py` where those
already cover the area, and in a new `test/python/test_plotting_timeseries.py`
for plotting. There are no C++ time series tests yet; C++ behavior that
Python cannot observe gets a new `test/test_timeseries.cpp`.

## Phase A: Domain accessor

- [x] **A1.** `Domain` and `TimeSeries.domain`.
  - [x] **A1a.** [2] (06ebd30e5) Bind the C++ `TimeSeries::domain()` and add the
    `stablebear.timeseries.Domain` named tuple and the `domain` property,
    converted to the series' unit. Export `Domain`.
    Tests: series built from observations gives first and last timestamps,
    both closed; after `to("ms")`, ends scaled by 1000; field access and
    tuple unpacking.
  - [x] **A1b.** [1] (9abe983a5) Tests for windows and channels (no new code expected).
    Tests: windows for `closed` = `"left"`, `"right"`, `"both"`,
    `"none"` have ends 0 and `size` with matching flags; a partial
    window's cut end and flag follow the series; two channels give the
    intersection, with the flags of the ends that bound it.
- [x] **A2.** [1] (b3163cb59) Point the `TimeSeries` docstring's prose on domains to the
  property; add `Domain` to `docs/api_timeseries.rst`.

## Phase B: Origin

- [x] **B1.** [3] (018d84cdb) C++ `Channel` origin: `m_origin` (default 0), `origin()`,
  `with_origin()`; kept by `take`, `until`, `with_start_closed`, and
  converted with the times when the channel changes unit; compared by
  `operator==`. Bind it and add `TimeSeries.origin` (float for one channel,
  `FloatTensor` for several).
  Tests: 0 for a series built from observations; one value per channel for
  several; read-only.
- [x] **B2.** `windows()` sets each window channel's origin to
  `channel.origin() + shift`.
  Note: once windows have non-zero origins, saving must keep them, because
  `operator==` compares origins and the round-trip tests compare tensors of
  windows. Saving therefore comes first, together with the epoch, so the
  file format changes once (moved here from C1 and D1).
  - [x] **B2a.** [2] (6d050da0e) C++ `Epoch` (an `int64_t` count and a `TimeUnit`) and
    `std::optional<Epoch>` on `Times`; `to` copies it, `operator==` compares
    it, `with_epoch` replaces it; `Epoch` can be built from a
    `std::chrono::sys_time`. Not bound to Python yet.
    Tests: none here; the epoch is tested through Python once it is bound
    (C2, C3, D1).
  - [x] **B2b.** [3] (9629137bd, d37757337) Write and read each channel's origin and its times' epoch
    in the version 3 record (`include/sbear/io/timeseries_io.hpp`), and
    replace the pre-release `test/golden/serialization/0.6-pre-timeseries`
    corpus, which no longer loads. Time series are unreleased, so this
    replaces the corpus rather than adding a version; confirm with the
    maintainer before deleting it.
    Tests: the existing round-trip tests still pass; origins are tested
    through Python in B2c, and epochs in D1.
  - [x] **B2c.** [2] (7709bc253) Windows set the origin.
    Tests: with no snapping, each window's origin equals its `return_bounds`
    start; `origin + t` is within a few ulps of the original timestamps;
    save/load and pickle keep the origins of a window tensor.
  - [x] **B2d.** [1] (76f270503) Snapping and nesting (no new code expected).
    Tests: a start that snaps onto an observation gives that observation's
    time; two channels that snap differently get different origins; a
    window of a window has its origin in the first series.
- [x] **B3.** [3] (846a7808d) Move the origin from `Channel` into `Times`, next to the
  epoch, so that every path that passes a channel's times along keeps both.
  Add `Times::with_values` and use it where windowing and delay embeddings
  build new timestamps; `Times::to` converts the origin.
  Note: element-wise operations rebuilt channels from their parts and
  dropped the origin; with the origin on `Times`, no such path can.
  Tests: kept by delay embeddings; converted exactly by `TimeSeries.to`;
  the existing origin and round-trip tests still pass.
- [x] **B4.** [3] (df765bf7e) `array_equal` ignores the origin, for series and for tensors
  of series: add `same_observations()` to `Channel` and `TimeSeries`, which
  skips the origin and epoch of the times, and use it in
  `TimeSeries.array_equal` (`src/python/py_timeseries_bind.hpp`) and, for
  elements that have it, in tensor `array_equal` (`src/python/py_tensor.hpp`).
  Note: with the origin and epoch both on `Times`, this also ignores the
  epoch, so C8 only adds its tests. Python cannot observe `operator==`; the
  round-trip tests compare origins directly instead.
  Tests: two windows with the same contents from different places are
  `array_equal` with different origins; two tensors of such windows are
  `array_equal`.
- [ ] **B5.** [2] Docs: `origin` docstring; `Window` docstring on origin versus
  the unsnapped bounds; a paragraph in `docs/timeseries_windows.rst`.

## Phase C: Epoch on `Times`

- [ ] **C1.** [1] Bind the C++ epoch on `Times` (B2a) to Python, internally;
  the public API follows in C2. (The C++ work moved to B2a.)
- [ ] **C2.** Python `Times(..., epoch=)` and `Times.epoch`.
  - [ ] **C2a.** [2] Accept and convert.
    Tests: `str`, `np.datetime64`, and naive `datetime` give the same
    instant, each at its own resolution; default `None`; `to` keeps it;
    `__eq__` compares count and resolution; `repr` shows it.
  - [ ] **C2b.** [1] Reject.
    Tests: timezone-aware `datetime` raises `ValueError`; `W`, `M`, `Y`
    resolutions raise; `NaT` and a non-date raise.
- [ ] **C3.** [1] `Times.with_epoch`.
  Tests: sets, replaces, and removes (`None`); timestamps and unit
  unchanged; invalid epochs raise as in the constructor.
- [ ] **C4.** [2] `Times` from a `datetime64` array.
  Tests: `datetime64[m]` with `unit="min"` gives the first stamp as epoch
  and exact whole minutes; with `unit="h"`, divided once; with an explicit
  epoch, offsets from it.
- [ ] **C5.** [3] Series epoch: `TimeSeries.epoch` (one value), and windowing
  copies the source channel's epoch into each window's `Times`.
  Tests: kept by windows, delay embeddings, `series[i]`, an element-wise
  operation, and `TimeSeries.to`; one value for a series with several
  channels.
- [ ] **C6.** [1] `from_channels` epoch rule.
  Tests: equal epochs kept; different epochs raise `ValueError`; some
  `None` and some set raise `ValueError`.
- [ ] **C7.** [2] `TimeSeries.with_epoch` (C++ and Python), sharing
  observations.
  Tests: sets every channel; values, times, and origins unchanged;
  `None` removes; works on a window.
- [ ] **C8.** [1] Tests that `array_equal` ignores the epoch, for series and
  tensors of series (the comparison from B4 already skips it).
  Tests: series differing only in epoch are `array_equal`, and so are
  tensors of them.

## Phase D: Saving origin and epoch

- [ ] **D1.** [1] Python tests that save/load and pickle keep an epoch set
  with `Times(..., epoch=)`, at `as` and `d` resolution, and that a series
  without an epoch loads without one. (Writing and reading moved to B2b.)

## Phase E: Dates as time points

- [ ] **E1.** [1] `TimeSeries(values, times=stamps)` with a `datetime64` array
  builds `Times(stamps, unit="s")`.
  Tests: unit `s`, epoch the first stamp, timestamps in seconds.
- [ ] **E2.** [2] `Times.to_datetime64(resolution="us")`.
  Tests: default `us`; `ns` and `s`; rounding to nearest; no epoch raises
  `ValueError` mentioning `with_epoch`.
- [ ] **E3.** Date queries.
  - [ ] **E3a.** [3] Scalar dates: `np.datetime64`, naive `datetime`, `date`,
    converted per channel as `(q - epoch) - origin`.
    Tests: each equals the numeric query at `q - epoch`; on a window,
    equals the parent series at that date.
  - [ ] **E3b.** [1] Arrays of `datetime64`.
    Tests: equals the numeric array query; shape preserved.
  - [ ] **E3c.** [2] Errors.
    Tests: no epoch raises `ValueError` mentioning `with_epoch`; `unit` with
    a date raises `TypeError`; a string raises; mixed dates and numbers
    raise `TypeError`; a timezone-aware `datetime` raises `ValueError`.
  - [ ] **E3d.** [2] Out-of-domain message in dates.
    Tests: the message names the date and the domain as dates.
  - [ ] **E3e.** [1] Snapping (no new code expected).
    Tests: a date that converts to just below a timestamp gets that
    observation.
- [ ] **E4.** Durations.
  - [ ] **E4a.** [2] `Window` size and step as `timedelta64` or `timedelta`,
    converted when applied.
    Tests: same windows as the numeric equivalent on a series in `min` and
    one in `h`; `repr` shows the duration; `M` and `Y` raise.
  - [ ] **E4b.** [1] `TimeDelayEmbedding` delay likewise.
    Tests: same embedding as the numeric delay on series in `min` and `h`;
    `repr`; `M` raises.
- [ ] **E5.** `TimeSeries.bounds`.
  - [ ] **E5a.** [2] Numeric bounds.
    Tests: equal to `domain` for a series built from observations; a
    window's ends are `origin` plus its domain ends with the same flags,
    and match its `return_bounds` row without snapping; one `Domain` per
    channel for several.
  - [ ] **E5b.** [1] `dates=True`.
    Tests: `epoch + origin + start/end` at `us` and at `s`; no epoch
    raises; window start dates match those computed from `return_bounds`
    and the epoch.
- [ ] **E6.** [3] Docs: a section on epochs, building `Times` from dates, date
  queries, durations, and `bounds` in `docs/timeseries.rst` or
  `docs/timeseries_units.rst`; docstrings of the new parameters.

## Phase F: `plot_timeseries`

- [ ] **F1.** [3] A scalar series on one axes: steps from the first
  observation to the domain end, `ax=None` uses the current axes, axis unit
  stored with `set_units`, label `time (<unit>)` if empty; returns the
  lines.
  Tests: x/y data equal the step vertices; a window extends its last
  value to `size`; a user-set label is kept.
- [ ] **F2.** [1] `markers=True` and `**kwargs` forwarding.
  Tests: markers at each observation; `color` and `label` reach the line.
- [ ] **F3.** Time axis rules.
  - [ ] **F3a.** [2] `unit=` and conversion into an existing axis unit.
    Tests: `unit="ms"` on a series in `s` scales by exactly 1000 and labels
    `time (ms)`; a series in `ms` after one in `s` is converted to `s`.
  - [ ] **F3b.** [2] Shared axes and conflicts.
    Tests: two `sharex` axes agree; `unit="ms"` on an axis in `s` raises
    `ValueError`; an axis with a date converter raises `TypeError`.
- [ ] **F4.** [1] Vector channels: one line per component.
  Tests: `(3,)` gives three lines labeled `x[0]`..`x[2]`; an explicit
  `label` is used as-is; `(2, 2)` raises `ValueError`.
- [ ] **F5.** [3] `TimeSeriesTensor` input and `relative`.
  Tests: three windows give three lines in successive cycle colors; by
  default each starts at its origin and matches the series' points there;
  with `relative=True` each starts at 0; an empty tensor draws nothing;
  a 2-D tensor raises.
- [ ] **F6.** Several channels as rows.
  - [ ] **F6a.** [2] `ax=None` creates `n_channels` rows sharing x.
    Tests: three rows, lines in channel order; x label on the last row
    only; channels in `s` and `ms` both drawn in `s`.
  - [ ] **F6b.** [1] `ax` as a sequence.
    Tests: three axes used in order; a single `Axes` and a sequence of two
    raise `ValueError` mentioning `series[i]`; a scalar channel next to a
    `(2,)` channel gives 1 and 2 lines.
- [ ] **F7.** Date axes for series with an epoch.
  - [ ] **F7a.** [3] Placement at `epoch + origin + t`, microsecond rounding.
    Tests: a series in `h` matches `mdates.date2num` of the expected dates;
    windows at their dates; `ns` times rounded to the nearest microsecond.
  - [ ] **F7b.** [2] Rules and errors.
    Tests: `relative=True` gives a numeric axis from 0; `unit` with an
    epoch raises unless `relative=True`; a series without an epoch on a
    date axis raises `TypeError`; one with an epoch on a unit axis raises
    `ValueError`; several channels make every row a date axis; two epochs
    on one date axis land at their own dates.
- [ ] **F8.** [1] Non-numeric dtypes raise `TypeError` naming the dtype.
  Tests: a point cloud series and a barcode series raise.
- [ ] **F9.** [5] Docs: "Time series" section in `docs/plotting.rst` with three
  figures (scalar series; two channels in different units; windows in
  place and with `relative=True`), light and dark images, and code
  dropdowns from `docs/_static/gen_plotting_gallery.py`; update
  `docs/api_plotting.rst` and point to it from `docs/timeseries.rst`.
  Rebuild the docs.

## Phase G: Tutorial and changelog

- [ ] **G1.** [3] Tides notebook: build `Times` from the stamps, plot the
  levels with `plot_timeseries`, and take window dates from
  `bounds(dates=True)`; drop `to_dates`. Re-run the notebook and rebuild the
  docs.
- [ ] **G2.** [1] Changelog entries for `domain`, `origin`, epochs, dates as
  time points, and `plot_timeseries`.
