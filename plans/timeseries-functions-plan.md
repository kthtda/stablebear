# Time series follow-ups: implementation plan

Progress: 0 of 10 items, 0 of 19 points (0%)

Specs: [timeseries-functions-spec.md](timeseries-functions-spec.md)
(phase A, #265) and `timeseries-point-clouds-spec.md`, written in B1
(phase B, #267). Phase C finishes #263, #264, and #266, which the
[pandas plan](pandas-timeseries-plan.md) implemented.

## Phase A: functions instead of classes (#265)

- [ ] **A1.** [3] `cut_windows` replaces `Window` in
  `stablebear/timeseries/window.py`, with its docstring, and in the
  tests, internal callers, golden generator, and figure scripts, without
  `return_bounds`; `TimeSeries.source_domain` gives where a window lies.
  Tests: the existing window tests call `cut_windows`, and those of bounds
  check each window's `source_domain`; `source_domain` of a series built
  from observations, a dated window, a series without a unit, and
  channels at different origins; the tests of `Window.size`,
  `Window.step`, and its `repr` become tests that durations of each type
  give the same windows.
- [ ] **A2.** [3] `delay_embed` replaces `TimeDelayEmbedding` in
  `stablebear/timeseries/embedding.py` likewise.
  Tests: as for A1, with `delay`.
- [ ] **A3.** [2] The guide pages and the tide gauge tutorial use
  `cut_windows` and `delay_embed`, and show where windows lie by their
  `source_domain` instead of `return_bounds`.

## Phase B: point clouds from time series (#267)

- [ ] **B1.** [1] Write `plans/timeseries-point-clouds-spec.md`, settling
  the open questions of #267: the name and signature, what a series of
  scalar observations gives, how channels combine, and whether "every
  observation" includes the start of the domain and the end of a window.
- [ ] **B2.** [3] The "every observation" mode: a series of vector
  observations gives a `PointCloud` of one point per time at which it
  changes, and a `TimeSeriesTensor`, such as windows, a `PointCloudTensor`
  whose clouds can differ in size.
  Tests: a delay embedding gives its distinct points; windows give clouds
  of their own sizes; the choices of B1 for scalars, channels, and ends.
- [ ] **B3.** [2] The uniform grid mode: `n` points, or points every
  `step`, over each series' domain from its time 0.
  Tests: `n` and `step` give the same cloud where they agree; windows of
  one size give clouds of one size; a step that does not divide the
  domain.
- [ ] **B4.** [1] The given-times mode: durations, or numbers for a series
  without a unit, as `evaluate` takes them.
  Tests: equals evaluating at those times; times outside the domain
  raise.
- [ ] **B5.** [2] A guide section on point clouds from time series, and
  the tide gauge tutorial builds its window clouds with it instead of
  evaluating on a grid by hand.

## Phase C: changelog and documentation upkeep (#263, #264, #266)

- [ ] **C1.** [1] `CHANGELOG.md` cites #263 and #264 in the time series
  entries, and describes `cut_windows` and `delay_embed` (#265) and point
  clouds from time series (#267).
- [ ] **C2.** [1] Turn on Sphinx's nitpicky mode, ignoring the private base
  classes and external classes that the API reference names, so that a
  broken link to the API reference (#266) fails the docs build.
  Tests: the docs build without warnings.
