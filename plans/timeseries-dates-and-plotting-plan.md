# Time series dates and plotting: implementation plan

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
  - [x] **A1a.** Bind the C++ `TimeSeries::domain()` and add the
    `stablebear.timeseries.Domain` named tuple and the `domain` property,
    converted to the series' unit. Export `Domain`.
    Tests: series built from observations gives first and last timestamps,
    both closed; after `to("ms")`, ends scaled by 1000; field access and
    tuple unpacking.
  - [x] **A1b.** Tests for windows and channels (no new code expected).
    Tests: windows for `closed` = `"left"`, `"right"`, `"both"`,
    `"none"` have ends 0 and `size` with matching flags; a partial
    window's cut end and flag follow the series; two channels give the
    intersection, with the flags of the ends that bound it.
- [ ] **A2.** Point the `TimeSeries` docstring's prose on domains to the
  property; add `Domain` to `docs/api_timeseries.rst`.

## Phase B: Origin

- [ ] **B1.** C++ `Channel` origin: `m_origin` (default 0), `origin()`,
  `with_origin()`; kept by `take`, `until`, `with_start_closed`, and
  converted with the times when the channel changes unit; compared by
  `operator==`. Bind it and add `TimeSeries.origin` (float for one channel,
  `FloatTensor` for several).
  Tests: 0 for a series built from observations; one value per channel for
  several; read-only.
- [ ] **B2.** `windows()` sets each window channel's origin to
  `channel.origin() + shift`.
  - [ ] **B2a.** Code and the basic case.
    Tests: with no snapping, each window's origin equals its `return_bounds`
    start; `origin + t` is within a few ulps of the original timestamps.
  - [ ] **B2b.** Snapping and nesting (no new code expected).
    Tests: a start that snaps onto an observation gives that observation's
    time; two channels that snap differently get different origins; a
    window of a window has its origin in the first series.
- [ ] **B3.** Origin propagation through operations that rebuild channels.
  Check each path; fix the ones that drop the origin.
  Tests: kept by delay embeddings, by an element-wise operation (persistent
  homology on a point cloud series), by `series[i]`, and by
  `from_channels`; converted exactly by `TimeSeries.to`.
- [ ] **B4.** `array_equal` ignores the origin: add a C++ comparison that
  skips it and bind `array_equal` to it instead of `operator==`
  (`src/python/py_timeseries_bind.hpp:67`). Add `test/test_timeseries.cpp`.
  Tests: two windows with the same contents from different places are
  `array_equal` with different origins; in C++, `operator==` is false for
  them and the new comparison is true.
- [ ] **B5.** Docs: `origin` docstring; `Window` docstring on origin versus
  the unsnapped bounds; a paragraph in `docs/timeseries_windows.rst`.

## Phase C: Epoch on `Times`

- [ ] **C1.** C++ `Epoch` (an `int64_t` count and a `TimeUnit`) and
  `std::optional<Epoch>` on `Times`; `to` copies it, `operator==` compares
  it, `with_epoch` replaces it. Bind them.
  Tests (C++): default empty; `to` keeps it; equality distinguishes
  epochs; `with_epoch` replaces and clears.
- [ ] **C2.** Python `Times(..., epoch=)` and `Times.epoch`.
  - [ ] **C2a.** Accept and convert.
    Tests: `str`, `np.datetime64`, and naive `datetime` give the same
    epoch; default `None`; `to` keeps it; `__eq__` compares it; `repr` shows
    it.
  - [ ] **C2b.** Reject.
    Tests: timezone-aware `datetime` raises `ValueError`; `W`, `M`, `Y`
    resolutions raise; `NaT` and a non-date raise.
- [ ] **C3.** `Times.with_epoch`.
  Tests: sets, replaces, and removes (`None`); timestamps and unit
  unchanged; invalid epochs raise as in the constructor.
- [ ] **C4.** `Times` from a `datetime64` array.
  Tests: `datetime64[m]` with `unit="min"` gives the first stamp as epoch
  and exact whole minutes; with `unit="h"`, divided once; with an explicit
  epoch, offsets from it.
- [ ] **C5.** Series epoch: `TimeSeries.epoch` (one value), and windowing
  copies the source channel's epoch into each window's `Times`.
  Tests: kept by windows, delay embeddings, `series[i]`, an element-wise
  operation, and `TimeSeries.to`; one value for a series with several
  channels.
- [ ] **C6.** `from_channels` epoch rule.
  Tests: equal epochs kept; different epochs raise `ValueError`; some
  `None` and some set raise `ValueError`.
- [ ] **C7.** `TimeSeries.with_epoch` (C++ and Python), sharing
  observations.
  Tests: sets every channel; values, times, and origins unchanged;
  `None` removes; works on a window.
- [ ] **C8.** `array_equal` also ignores the epoch (extend the B4
  comparison).
  Tests: series differing only in epoch are `array_equal`; C++
  `operator==` is false for them.

## Phase D: Saving origin and epoch

- [ ] **D1.** Write and read each channel's origin and its times' epoch in
  the version 3 record (`include/sbear/io/timeseries_io.hpp`), and replace
  the pre-release `test/golden/serialization/0.6-pre-timeseries` corpus,
  which no longer loads. Time series are unreleased, so this replaces the
  corpus rather than adding a version; confirm with the maintainer before
  deleting it.
  Tests: save/load and pickle keep the origins of a window tensor; keep an
  epoch exactly at `as` and `d` resolution; a series without an epoch
  loads without one.

## Phase E: Dates as time points

- [ ] **E1.** `TimeSeries(values, times=stamps)` with a `datetime64` array
  builds `Times(stamps, unit="s")`.
  Tests: unit `s`, epoch the first stamp, timestamps in seconds.
- [ ] **E2.** `Times.to_datetime64(resolution="us")`.
  Tests: default `us`; `ns` and `s`; rounding to nearest; no epoch raises
  `ValueError` mentioning `with_epoch`.
- [ ] **E3.** Date queries.
  - [ ] **E3a.** Scalar dates: `np.datetime64`, naive `datetime`, `date`,
    converted per channel as `(q - epoch) - origin`.
    Tests: each equals the numeric query at `q - epoch`; on a window,
    equals the parent series at that date.
  - [ ] **E3b.** Arrays of `datetime64`.
    Tests: equals the numeric array query; shape preserved.
  - [ ] **E3c.** Errors.
    Tests: no epoch raises `ValueError` mentioning `with_epoch`; `unit` with
    a date raises `TypeError`; a string raises; mixed dates and numbers
    raise `TypeError`; a timezone-aware `datetime` raises `ValueError`.
  - [ ] **E3d.** Out-of-domain message in dates.
    Tests: the message names the date and the domain as dates.
  - [ ] **E3e.** Snapping (no new code expected).
    Tests: a date that converts to just below a timestamp gets that
    observation.
- [ ] **E4.** Durations.
  - [ ] **E4a.** `Window` size and step as `timedelta64` or `timedelta`,
    converted when applied.
    Tests: same windows as the numeric equivalent on a series in `min` and
    one in `h`; `repr` shows the duration; `M` and `Y` raise.
  - [ ] **E4b.** `TimeDelayEmbedding` delay likewise.
    Tests: same embedding as the numeric delay on series in `min` and `h`;
    `repr`; `M` raises.
- [ ] **E5.** `TimeSeries.bounds`.
  - [ ] **E5a.** Numeric bounds.
    Tests: equal to `domain` for a series built from observations; a
    window's ends are `origin` plus its domain ends with the same flags,
    and match its `return_bounds` row without snapping; one `Domain` per
    channel for several.
  - [ ] **E5b.** `dates=True`.
    Tests: `epoch + origin + start/end` at `us` and at `s`; no epoch
    raises; window start dates match those computed from `return_bounds`
    and the epoch.
- [ ] **E6.** Docs: a section on epochs, building `Times` from dates, date
  queries, durations, and `bounds` in `docs/timeseries.rst` or
  `docs/timeseries_units.rst`; docstrings of the new parameters.

## Phase F: `plot_timeseries`

- [ ] **F1.** A scalar series on one axes: steps from the first
  observation to the domain end, `ax=None` uses the current axes, axis unit
  stored with `set_units`, label `time (<unit>)` if empty; returns the
  lines.
  Tests: x/y data equal the step vertices; a window extends its last
  value to `size`; a user-set label is kept.
- [ ] **F2.** `markers=True` and `**kwargs` forwarding.
  Tests: markers at each observation; `color` and `label` reach the line.
- [ ] **F3.** Time axis rules.
  - [ ] **F3a.** `unit=` and conversion into an existing axis unit.
    Tests: `unit="ms"` on a series in `s` scales by exactly 1000 and labels
    `time (ms)`; a series in `ms` after one in `s` is converted to `s`.
  - [ ] **F3b.** Shared axes and conflicts.
    Tests: two `sharex` axes agree; `unit="ms"` on an axis in `s` raises
    `ValueError`; an axis with a date converter raises `TypeError`.
- [ ] **F4.** Vector channels: one line per component.
  Tests: `(3,)` gives three lines labeled `x[0]`..`x[2]`; an explicit
  `label` is used as-is; `(2, 2)` raises `ValueError`.
- [ ] **F5.** `TimeSeriesTensor` input and `relative`.
  Tests: three windows give three lines in successive cycle colors; by
  default each starts at its origin and matches the series' points there;
  with `relative=True` each starts at 0; an empty tensor draws nothing;
  a 2-D tensor raises.
- [ ] **F6.** Several channels as rows.
  - [ ] **F6a.** `ax=None` creates `n_channels` rows sharing x.
    Tests: three rows, lines in channel order; x label on the last row
    only; channels in `s` and `ms` both drawn in `s`.
  - [ ] **F6b.** `ax` as a sequence.
    Tests: three axes used in order; a single `Axes` and a sequence of two
    raise `ValueError` mentioning `series[i]`; a scalar channel next to a
    `(2,)` channel gives 1 and 2 lines.
- [ ] **F7.** Date axes for series with an epoch.
  - [ ] **F7a.** Placement at `epoch + origin + t`, microsecond rounding.
    Tests: a series in `h` matches `mdates.date2num` of the expected dates;
    windows at their dates; `ns` times rounded to the nearest microsecond.
  - [ ] **F7b.** Rules and errors.
    Tests: `relative=True` gives a numeric axis from 0; `unit` with an
    epoch raises unless `relative=True`; a series without an epoch on a
    date axis raises `TypeError`; one with an epoch on a unit axis raises
    `ValueError`; several channels make every row a date axis; two epochs
    on one date axis land at their own dates.
- [ ] **F8.** Non-numeric dtypes raise `TypeError` naming the dtype.
  Tests: a point cloud series and a barcode series raise.
- [ ] **F9.** Docs: "Time series" section in `docs/plotting.rst` with three
  figures (scalar series; two channels in different units; windows in
  place and with `relative=True`), light and dark images, and code
  dropdowns from `docs/_static/gen_plotting_gallery.py`; update
  `docs/api_plotting.rst` and point to it from `docs/timeseries.rst`.
  Rebuild the docs.

## Phase G: Tutorial and changelog

- [ ] **G1.** Tides notebook: build `Times` from the stamps, plot the
  levels with `plot_timeseries`, and take window dates from
  `bounds(dates=True)`; drop `to_dates`. Re-run the notebook and rebuild the
  docs.
- [ ] **G2.** Changelog entries for `domain`, `origin`, epochs, dates as
  time points, and `plot_timeseries`.
