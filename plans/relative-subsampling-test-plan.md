# Relative subsampling and distributions: test plan

This plan covers the behavior this branch adds or changes for #243: relative
subsampling, the distributions API, partial-sample policies, generator
validation, and the distance-weight heatmap. Work through it in order. Every
item depends only on items above it, so each new test builds on behavior that
earlier tests have already confirmed. Every input and expected value below
was checked against the x86-64 Linux build at `4e5c128bb`.

## How to work through this plan

Read [TESTING.md](../test/TESTING.md) before writing any tests. Each item is
one review unit: one behavior, adding at most one or two simple tests or a
single statistical test. Before implementing an item, confirm its example and
check whether existing tests already cover it. Keep new work uncommitted for
review, and mark an item complete only once we have reviewed its coverage,
including a decision that existing tests suffice.

Each item gives its input, the expected result, and **Catches:**, the bug that
makes the test fail. That answers the review question: **What bug would make
this test fail, and can I see that immediately?** Prefer public Python
behavior. Tags: **[S]** statistical (see the conventions), **[C++]** a
guarantee the public API cannot reach, **(existing)** confirm existing
coverage without adding a test, **(decision)** a choice to make before the
items that follow, and **(optional)** safe to skip without losing important
coverage. Check off a skipped optional item with the note "skipped". The plan
is finished when every item without (optional) is checked.

Milestones: once item 27 is checked, relative sampling is confirmed to draw
from the right distributions; once item 35 is checked, both samplers also
apply every partial policy to the right eligible points. The coverage map
near the end lists the items that check each part of the log-space sampling
code. Build and test commands are in [AGENTS.md](../AGENTS.md#testing).

## Conventions

- **Files.** Distribution tests go in a new
  `test/python/test_distributions.py`, relative-sampling tests in
  `test/python/test_subsample_relative.py`, uniform `subsample` policy tests in
  `test/python/test_subsample.py`, and heatmap tests in
  `test/python/test_plotting.py`, using its `ax` fixture. C++ items name their
  file.
- **Deterministic checks** use no seed. Every item without [S] is one, except
  items 38-40 and 43, which seed generators to compare runs or streams with
  each other, never with recorded outputs. Deterministic checks rely on a
  unique eligible point, on drawing every eligible point, or on an
  overwhelming weight ratio; state the failure probability in a comment when
  it is not exactly zero.
- **[S] with replacement:** draw one sample of 20000 points
  (`n_points=20000`) and compare
  `np.bincount(np.asarray(samples.indices[0, 0]), minlength=k) / 20000` with
  the exact probabilities within the tolerance below. Where an item checks a whole
  law rather than a few points, use the reference `np.linspace(-1.5, 3, 10)`
  (asymmetric around a query at 0) and evaluate the expected formula on it.
  **[S] without replacement:** item 20 compares how often each point is
  included in 5000 samples with how often NumPy includes it in 5000 samples
  of its own, within five standard errors of the difference. Item 21 reads each sample's draws in order, with `atol=0.03`.
- **Tolerance.** With replacement, each frequency must lie within five of its
  own standard errors, `5 * sqrt(p * (1 - p) / n_draws)`, so points with
  probability 0 must never be drawn. This needs an expected count
  `p * n_draws` of at least 30 wherever `p` is nonzero; choose inputs
  accordingly (the test asserts it). Each wrong law an item lists must miss
  some cell by at least three times that cell's tolerance with 10 points and
  20000 draws (item 20 states its own sizes). For item 21, the standard error is
  at most `sqrt(0.25 / 6000) = 0.0065`, so `atol=0.03` is more than 4.5
  standard errors, and each listed wrong law must be at least 0.08 away in
  some cell.
  Record the margins in the plan item, not in a test comment.
- **Fast and thorough modes.** Every [S] test runs in two modes. The fast
  case runs with the unit tests and uses the sizes above. The thorough case
  is a `pytest.param(..., marks=pytest.mark.statistical)` with a larger
  input and far more draws, so the same assertions catch smaller deviations;
  it is skipped unless `SB_RUN_STATISTICAL=1` is set, and CI runs it on the
  self-hosted runner (`statistical.yaml`). Parametrize over the sizes, for example
  `(n_reference, n_draws)` with `(10, 20_000)` and a thorough
  `(200, 10_000_000)`, as item 15 does.
  Express tolerances in standard errors so they tighten with the draw count;
  item 21 therefore uses `atol=0.03 * sqrt(6000 / n_samples)`.
- **Seeds.** Statistical checks pass `generator=sb.random.Generator(42)` and
  assert frequencies within tolerance, never exact draws: seeded outputs
  depend on the standard library and the `long double` width, which differ
  across the CI platforms. Do not rely on adjacent seeds being independent
  (see the follow-ups).
- **Distribution parameters.** Write every distribution parameter out, as
  in `Uniform(0, 0.5)` and `Gaussian(0, 1)`, unless the test checks a
  default.
- **Messages.** Match messages with `re.escape(...)` when they contain
  parentheses, brackets, or `+`; pasted raw into `match=`, they do not match.
- **Speed.** Read `samples.indices` once into a local variable; each access
  rebuilds every cell, about 14 ms for 6000 cells. A check takes about 3 ms
  with replacement and 0.2 s without, mostly extracting index arrays.
- **Replacement.** Both samplers default to `replace=True`. Items that draw
  without replacement pass `replace=False` explicitly.
- **Portability.** Some CI platforms use a 64-bit `long double`. Keep
  `((distance - mean) / std)**2` finite in double, and compare floating-point
  results with `pytest.approx` unless they are exact in double.

## Density values

These go in the new `test/python/test_distributions.py`; the first item
creates it. Gaussian and Uniform use one implementation for `.weight()` and
sampling, so items 1, 2, and 6 also guard the sampler; `Mixture.weight()` does
not.

- [x] **1. Gaussian absolute density.**
  `Gaussian(2, 0.5).weight([2, 2.5, 1.5, 3])` is
  `[0.7978845608, 0.4839414490, 0.4839414490, 0.1079819330]`, with peak
  `1/(0.5*sqrt(2*pi))`. **Catches:** a missing normalization constant, swapped
  mean and std, the variance used as the scale, and an exponent linear in the
  distance (0.2935 at 3, two standard deviations away).
- [x] **2. Bounded Uniform density.** `Uniform(-1, 1).weight([-1.5, -1, 0, 1])`
  is `[0, 0.5, 0.5, 0]`. Sampling uses the logarithm of this weight, so later
  sampling examples rely on these endpoints: the start is included and the end
  excluded. **Catches:** unit weight instead of `1/width`, a closed upper end,
  an open lower end, and rejected negative values.
- [x] **3. Mixture coefficients are normalized at construction.** `[1, 3]`
  gives `.coefficients == (0.25, 0.75)`; `coefficients=[2, 6]` by keyword gives
  the same; `[1e308, 1e308]` gives `(0.5, 0.5)`. Every later mixture example
  relies on this. **Catches:** unnormalized or overflowing coefficients, and a
  broken keyword. On x86-64 the 1e308 case checks the contract only, because
  the `long double` sum cannot overflow.
- [x] **4. Equality and hashing.** Two small tests. Equal:
  `Gaussian(mean=2, std=0.5) == Gaussian(2, 0.5)`, `Uniform() == Uniform(0, 1)`,
  and `Mixture([Uniform(0, 1), Uniform(5, 6)], [1, 3])` equals the same mixture
  with coefficients `[2, 6]`, with equal hashes. Unequal:
  `Gaussian(2, 0.5) != Gaussian(0.5, 2)` and `Gaussian(0, 1) != 0`. This
  also pins the Gaussian keyword names and the `Uniform()` default of
  `[0, 1)`.
  **Catches:** identity-based equality, equality that ignores parameter values
  or their order, raw instead of normalized coefficients, hashes inconsistent
  with equality, and comparison with a non-distribution raising instead of
  returning unequal.
- [x] **5. Mixture weights are elementwise.** For
  `Mixture([Uniform(0, 2), Uniform(10, 11)], [1, 3])`, values `[0, 1, 2]` give
  `[0.125, 0.125, 0]`. **Catches:** normalizing over the supplied values, or an
  absent component redistributing its coefficient (both give
  `[0.5, 0.5, 0]`). Sampling uses a separate log-space implementation, checked
  later.
- [x] **6. Unbounded Uniform unit weight.**
  `Uniform(1, inf).weight([0.5, 1, 1e300])` is `[0, 1, 1]`. **Catches:**
  `1/inf = 0` on unbounded support.

## Queries and output

- [x] **7. Review the existing coordinate-query test.**
  `test_coordinate_queries_select_points_from_their_own_regions` in
  `test/python/test_subsample_relative.py` uses reference coordinates 0, 4,
  and 9, external queries 3 and 8, and `Uniform(0, 1.5)`, which select 4 and
  9 respectively. Each region contains exactly one eligible point, so no seed
  is needed. Review it against TESTING.md and keep or adjust it.
  **Catches:** wrong regions, and compact positions returned instead of source
  indices (point 4 is source index 1 but compact position 0).
- [x] **8. The default keeps every eligible point.** Reference coordinates 0,
  1, 4; coordinate query `[[0.5]]`; `Uniform(0, 1.5)`; `n_points=3` without
  replacement; `allow_partial` omitted. Expect exactly indices {0, 1} and
  sample shape `(2, 1)`. The distances 0.5, 0.5, and 3.5 touch no interval
  endpoint. Later items that request more points than are eligible rely on
  this. **Catches:** a default of `"no"` (ValueError), and short samples that
  are padded or rejected.
- [x] **9. Index queries keep order and repetition.** Reference coordinates 0,
  4, 9 with `Uniform(0, 0.5)` and `n_points=1`, so each query selects only
  itself. Signed queries `[-1, 0, -3, -1]` select 9, 0, 0, 9. A second test
  shows that `query=None` gives three queries selecting 0, 4, 9.
  **Catches:** negative indices not counted from the end, a bound that rejects
  `-3` (minus the population), sorted or deduplicated queries, and `None` not
  meaning every reference point in order.
- [x] **10. Strided unsigned index tensors.**
  `sb.IntTensor([2, 1, 0, 1], dtype=...)[::2]`, parametrized over uint32 and
  uint64, selects 9, 0 with item 9's reference and distribution (coordinates
  0, 4, 9; `Uniform(0, 0.5)`) and `n_points=1`. **Catches:** reading the
  backing tensor instead of the view, and a broken conversion path for either
  unsigned type.
- [x] **11. Query, distribution, and sample axis order.** Reference coordinates
  0, 4, 9; queries `[0, 2]`; distributions
  `[Uniform(0, 0.5), Uniform(3.5, 5.5)]`; `n_points=1`; `n_samples=2`. Expect
  shape `(2, 2, 2)`. The first query selects 0, then 4; the second selects 9,
  then 4 (at distance 5); each sample repeats its cell's unique eligible
  point. **Catches:** swapped query and distribution axes (cell (0, 1) would
  hold 9 and cell (1, 0) would hold 4), distributions applied out of list
  order (4, then 0 for the first query), and source indices left over from the
  previous distribution.
- [x] **12. Output shapes.** One parametrized test with item 9's reference and
  distribution, `n_points=1`, and `n_samples=3`: a single distribution with
  queries `[0, 1]` gives `(2, 3)`; a one-element list gives `(2, 1, 3)`; an
  empty query `[]` gives `(0, 3)`, or `(0, 1, 3)` with a one-element list.
  **Catches:** a one-element list losing its axis, and empty queries dropping
  axes.
- [x] **13. Each reference kind and precision reaches its own binding.** One
  test with the `make_pcloud_or_distmat_tensor` fixture (pcloud32, pcloud64,
  distmat32, distmat64): build the reference from coordinates 0, 4, 9 with
  `make_pcloud_or_distmat_tensor(...)[()]`, then sample with query `[2, 0]`,
  `Uniform(0, 0.5)`, and `n_points=1`. Expect indices `[2]`, then `[0]`, and
  `samples.dtype` equal to the fixture tensor's dtype. **Catches:** a missing
  or misnamed 32-bit binding, 32-bit samples returned as 64-bit, a reference
  kind dispatched to the wrong binding, and matrix weights taken from the
  query position's row instead of the query index's (the matrix cases would
  select `[0]`, then `[1]`).

## Drawing from the right distributions

Sampling evaluates log weights, normalizes each query/distribution row over
the reference points, and treats points with a finite log weight as eligible.
Mixture coefficients sum to one at each level, components are normalized
densities (unbounded Uniform intervals use unit weight), and sampling
normalizes only the combined weights over the reference points. Use explicit
probabilities, not normalized `.weight()` output, as expected values.

- [x] **14. Euclidean distance in two dimensions.** Reference points `(3, 4)`,
  `(4.5, 4.5)`, and `(6, 0)` lie at Euclidean distances 5, 6.36, and 6 from the
  coordinate query `(0, 0)`. With `Uniform(0, 5.5)` and `n_points=3`, expect
  only index 0. **Catches:** another metric: Chebyshev distance or the first
  coordinate alone selects 0 and 1, while L1 or squared distance selects
  nothing. No other sampling example distinguishes these metrics: the rest
  are one-dimensional, and item 72's eligible points lie on the coordinate
  axes.
- [x] **15. With-replacement draws follow the weights.** [S] Reference
  `np.linspace(-1.5, 3, 10)`, coordinate query `[[0.0]]`, `Gaussian(0, 1)`.
  Expect frequencies proportional to `exp(-x**2 / 2)`. This is the first
  statistical test and sets the layout for the others: it is parametrized by
  `(distribution, density, reference_range)`, where `density` is the
  expected formula as a function of distance and `reference_range` the
  linspace's endpoints, and items 16-19 each add a case to it, with inputs
  adapted to a linspace and their wrong laws rechecked against it. The
  Gaussian cases of items 15 and 16 keep `(-1.5, 3)`, since farther points
  have expected counts below 30; the other cases use a wider range to include
  points outside a bounded support. **Catches:** a CDF built from log values
  instead of probabilities, uniform selection (58 tolerances away in its worst
  cell), a sign error in the exponent (398), a reversed CDF (38), an exponent
  linear in the distance (22), and a CDF shifted by one cell (7, the
  nearest).
- [x] **16. Gaussian mean and std in sampling.** [S] A case
  `Gaussian(mean=1.5, std=0.5)` in item 15's test, with expected density
  `exp(-((x - 1.5) / 0.5)**2 / 2)`. On item 15's linspace, the original
  `Gaussian(2, 2)` left dividing by the variance only 2.4 tolerances away, so
  this item uses these parameters instead; the smallest expected count is 52.
  **Catches:** a sampling exponent that ignores the mean (219 tolerances away
  in its worst cell) or adds it (520), ignores the std (24) or divides by the
  variance (12), and an exponent linear in the standardized distance (6.2, the
  nearest). Item 1 checks these parameters through `.weight()`; this item
  checks the `log_weight` path that sampling uses.
- [x] **17. Mixture law in sampling.** [S] A case
  `Mixture([Uniform(0, 0.5), Uniform(0, 3)], [1, 3])` in item 15's test, with
  reference range `(-1, 8)`, whose 10 points lie at distances 1, 0, 1, 2, and
  3 through 8. Distance 0 has weight `1/4 * 2 + 3/4 * 1/3 = 3/4`, distances in
  `[0.5, 3)` have `1/4`, and distances from 3 on have none, so expect `1/2` at
  distance 0, `1/6` at distances 1 and 2, and 0 elsewhere. Distance 3 sits on
  an excluded end; the smallest nonzero expected count is 3333. **Catches:**
  normalizing each component over the reference points (3.5 tolerances away
  in its worst cell, the nearest), a maximum instead of a log-sum (5.7),
  swapped operands in `log_add`'s exponent (9.4), ignored coefficients (11),
  `c` added instead of `log(c)` (6.1), ignored densities (11), and a missing
  `-inf` guard in `log_add` (NaN weights from distance 3 on).
- [x] **18. Nested mixture with an empty inner component.** [S] A case
  `Mixture([Mixture([Uniform(0, 1), Uniform(10, 11)], [1, 9]), Uniform(2, 3)], [1, 3])`
  in item 15's test, with item 17's reference range `(-1, 8)`. The weights
  are `1/4 * 1/10` at distances below 1, `3/4` at distances in `[2, 3)`, and 0
  elsewhere, so expect `1/31` at distance 0, `30/31` at distance 2, and 0
  elsewhere. Distances 1 and 3 sit on excluded ends and 2 on an included
  start; the smallest nonzero expected count is 645. This is the only check of
  nested mixtures in the log path. **Catches:** an empty component
  redistributing its coefficient, renormalizing over supported components, or
  raw inner coefficients (all 35 tolerances away in their worst cell), ignored
  inner coefficients (18), and ignored outer coefficients (9.4, the nearest).
- [x] **19. Gaussian density scale inside a sampling mixture.** [S] A case
  with the equal mixture of `Gaussian(0, 3)` and `Uniform(8, 10)` in item
  15's test, with item 17's reference range `(-1, 8)`. The expected density
  is `1/2 * gaussian(x, 0, 3) + 1/2 * uniform(x, 8, 10)`, so the Gaussian alone covers distances 0 through 7 and both components
  cover distance 8, the included start of the interval; expect `0.4233` at
  distance 8 and `0.1117` at distance 0. With the plan's earlier std of 2,
  distances near 8 fall below 30 expected draws, so this item uses a std of
  3; the smallest expected count is 147. **Catches:** a
  sampling log weight without `1/std` (13 tolerances away in its worst cell),
  without `sqrt(2*pi)` (11), or with the variance in the constant (15), and a
  Uniform weight without `1/width` (9.8, the nearest). Item 1 checks the
  density behind `.weight()`; this item checks `log_weight`, which could lose
  its constants unnoticed because they cancel for a single Gaussian.
- [x] **20. Draws without replacement follow the weights.** [S] A test laid
  out like item 15's: coordinate query `[[0.0]]`, `replace=False`,
  parametrized by `(distribution, density, reference_range)` and by
  `(n_reference, n_samples, n_points)`: a population of 100 points with 5000
  samples of 10 (about 0.3 s per case, within a 0.35 s budget), and a
  thorough 200 points with 120000 samples of 1, 10, 50, and 150 (6 to 15 s
  per case). Order within a sample is not checked: the test counts the
  fraction of samples that include each point and compares it with the
  fraction of as many `np.random.Generator.choice(..., replace=False, p=p)`
  samples that do, in `numpy_inclusion_frequencies`. NumPy's choice draws
  one point at a time in proportion to the remaining weights, the same law;
  this is not documented, but it matched the exact law in a check. An
  `assert_histograms_agree` helper allows five standard errors of the
  difference, `5 * sqrt(2 * q * (1 - q) / n)` with `q` the average of the two
  frequencies, and requires 30 samples on each side of every bin with
  `0 < q < 1`. Cases: `Gaussian(0, 1)` on
  `(-1.5, 2)` (smallest inclusion probability 0.022), and the equal mixture
  of `Uniform(0, 1)` and `Uniform(1, 3)` on the descending range
  `(5.5, -1.25)`, whose first 37 points are ineligible, so compaction shifts
  every eligible weight; 150 draws exceed the thorough case's eligible
  points, so that case checks that every eligible point is kept.
  **Catches** (tolerances away in the worst cell for the fast cases, from
  simulating each wrong algorithm; best case named): compacted weights out of
  line with their source indices (3.6, mixture), a reversed key comparison
  (34, Gaussian), keys `clock * weight` instead of `clock / weight` (21,
  Gaussian), raw clocks compared instead of their logarithms (4.2,
  Gaussian), uniform selection among eligible points (5.3, Gaussian), and an
  ineligible point drawn (it fails the count requirement).
- [x] **21. The second draw follows the remaining weights.** (existing)
  Covered by item 20, which compares how often each point is included in
  samples without replacement with NumPy's. Of the wrong laws an earlier version of this item
  listed, item 20 catches all but renormalizing each mixture component after
  every draw, which the sampler cannot do: it computes each log weight once
  and draws a whole sample from one sort of keys.
- [x] **22. One eligible point fills a sample with replacement.** Reference
  coordinates 0, 4, 9; query `[1]` (the middle point); `Uniform(0, 1)`;
  `n_points=3`; `replace=True`; parametrize `allow_partial` over `"no"` and
  omitted (the default `"keep"`). Expect `[1, 1, 1]` in both cases.
  **Catches:** an inverse-CDF index shifted by one onto a zero-weight
  neighbor, `"no"` demanding `n_points` eligible points with replacement, and
  the default returning only the eligible count with replacement (`[1]`).
  Only item 23 catches a switch to `lower_bound`.
- [x] **23. CDF boundary targets.** [C++] In `test/test_subsample.cpp`
  (already part of `sb_test`), include `<sbear/sampling/weighted_draw.hpp>` and
  call `index_for_target<double>` on the CDF `[0, 0.25, 0.25, 1, 1]`, from
  probabilities `[0, 1/4, 0, 3/4, 0]`; the explicit template argument lets a
  `std::vector<double>` convert to `std::span<const double>`. Target 1 maps
  to index 3 and target 0 to index 1.
  **Catches:** removing the guard for a uniform draw rounded up to the total
  (LWG 2524), which returns index 5, past the end; and switching to
  `lower_bound`, which selects the zero-weight entry 0. Public draws cannot
  reliably hit the rounded endpoint, and a separate cumulative-array test adds
  nothing beyond this and item 22.
- [x] **24. A far query keeps an eligible point.** Reference coordinates 0 and
  1, coordinate query `[[1000.0]]`, `Gaussian(0, 1)`, `n_points=1`,
  `allow_partial="no"`; parametrize `replace`. Expect index 1 in both modes.
  Index 0 has probability `exp(-999.5)`: without replacement it can never be
  drawn first, and with replacement only a uniform draw of exactly 0 selects
  it (about `2**-64`). **Catches:** normalizing without first shifting by the
  row maximum (every probability underflows, leaving no eligible point), a CDF
  built from unnormalized values (an out-of-range index), and a reversed key
  comparison without replacement (index 0). The query must be more than about
  151 standard deviations away, where `exp` underflows even in `long double`.
- [x] **25. Tiny positive weights stay eligible.** Reference coordinates 0 and
  10000, query `[0]`, `n_points=2` without replacement, `allow_partial="no"`,
  and the equal mixture of `Gaussian(0, 1)` and `Uniform(0, 1)`. Expect both
  indices, compared as a set. **Catches:** eligibility decided from linear
  weights or from `exp(log weight) > 0` (`exp(-5e7)` is zero even in
  `long double`), and `Mixture::log_weight` computed as `log(weight)`. Item 26
  covers far-tail eligibility for a plain Gaussian.
- [x] **26. Equal far-tail weights keep a random order.** [S] Reference
  coordinates 0, -1e20, and 1e20; query `[0]`; `Gaussian(0, 1)`;
  `n_points=2` without replacement; `n_samples` of 2000 plus a thorough
  120000 (4.5 s). Expect every first draw to be 0, and the second draws to
  be 1 and 2 with probability 1/2 each, using `assert_histogram_matches_exact`.
  **Catches:** replacing the difference comparator with precomputed keys
  `log(clock) - log_weight`: at a log weight of `-5e39` the clock is lost to
  rounding, ties fall back to index order, and the second draw is always 1
  (frequency 1 against 0.5, 8.9 tolerances away with 2000 samples). 1e20 is
  large enough for every `long double` width on the CI platforms and keeps
  `(distance / std)**2` finite in double.
- [x] **27. Infinite matrix distances have zero weight.** A distance matrix
  with rows `[0, 1, inf, inf]`, `[1, 0, inf, inf]`, `[inf, inf, 0, 2]`, and
  `[inf, inf, 2, 0]`; query `[0]`; `n_points=4` without replacement and
  `allow_partial="keep"`;
  parametrize the distribution over `Gaussian(0, 1)` and `Uniform(0, inf)`. Expect
  exactly indices {0, 1}. One precision and one replacement mode suffice.
  **Catches:** infinite distances reaching `Gaussian::log_weight`
  (OverflowError) and points at infinite distance counted as eligible (only
  the Gaussian case catches these, because `Uniform(0, inf)` excludes infinity
  at its open end), and an unbounded Uniform whose sampling weight loses the
  unit weight on its support (the Uniform case would select nothing). This is
  the only sampling check of an unbounded Uniform.

## Eligibility and partial policies

Items 28-32 cover `subsample_relative`; items 33-37 cover uniform `subsample`
in `test/python/test_subsample.py`. Both samplers share `_partial_policy` and
the native policy helpers, so item 33's `False` alias and item 35 are not
repeated for `subsample_relative`.

- [x] **28. "no" names the failing pair.** Reference coordinates 0, 1, 4;
  queries `[0, 2]`; distributions `[Uniform(0, 1.5), Uniform(0, 4.5)]`;
  `n_points=2` without replacement; `allow_partial="no"`. Only the second
  query's first distribution has too few eligible points (one). Expect
  ValueError "insufficient positive support for query 1, distribution 0".
  **Catches:** naming the reference index (2) instead of the query position,
  swapping the query and distribution positions, and counting the population
  instead of eligible points (no error).
- [x] **29. "no" with replacement raises only when no point is eligible.**
  Reference coordinates 0, 4, 9; query `[1]`; `Uniform(1, 2)`, which leaves no
  eligible point; `n_points=3`; `replace=True`; `allow_partial="no"`. Expect
  ValueError "insufficient positive support for query 0, distribution 0".
  Item 22 covers the one-eligible-point case. **Catches:** replacement
  skipping the check.
- [x] **30. Empty cells keep their axes.** With `allow_partial="keep"`:
  reference coordinates 0 and 4, query `[0]`, distributions
  `[Uniform(0, 0.5), Uniform(10, 11)]`, `n_points=2`, `n_samples=2`, and
  `replace=False` (with replacement the first cell would fill to two) give
  shape `(1, 2, 2)`; both samples of the first cell hold one point, and both
  of the second are empty with shape `(0, 1)`. Item 8 covers a partial sample
  under the default. **Catches:** an empty cell that raises or loses its axis,
  and short samples padded to `n_points`.
- [x] **31. "drop" applies even with replacement.** The documented example in
  `docs/subsampling_options.rst`: reference coordinates 0, 1, 4; queries
  `[0, 2]`; `Uniform(0, 1.5)`; `n_points=2`; `allow_partial="drop"`;
  parametrize `replace`. Expect the first query's sample to hold two draws
  from {0, 1}, exactly at the threshold, and the second query's sample to be
  empty. **Catches:** replacement bypassing the threshold (the second query
  would get `[2, 2]`), and an off-by-one that drops a sample at exactly
  `n_points`.
- [x] **32. The threshold precedes duplicate removal.** Reference
  `[[0], [0], [5]]`, coordinate query `[[0]]`, `Uniform(0, 1)`,
  `n_points=2`, `replace=False`, `allow_partial="no"`,
  `discard_duplicates=True`. Expect one
  point, `[[0.0]]`, and no error, as the docstring promises. **Catches:**
  eligibility counted after duplicate removal (`"no"` would raise), and
  duplicates kept (two points). Matrix duplicate rules need no relative test:
  both samplers call the same `discard_sample_duplicates` helper, which
  `test_subsample.py` and `test/test_subsample.cpp` cover.
- [x] **33. The uniform default keeps an undersized input, and False means
  "no".** Two small tests on a two-point cloud with `n_points=3` and
  `replace=False`. With
  `allow_partial` omitted, `subsample` returns one sample holding both points;
  with `allow_partial=False`, it raises ValueError "n_points exceeds the number
  of input points". `True` (the `above-population-partial` case of
  `test_sample_sizes_for_single_input`) and explicit `"no"`
  (`test_undersized_last_input_does_not_advance_generator`) are already
  covered. **Catches:** a default of `"no"`, and the `False` alias mapping to
  `"keep"`.
- [x] **34. Uniform "drop" on ragged input.** A `PointCloudTensor` whose
  elements hold 0, 2, 3, and 4 points; `n_points=3`; `allow_partial="drop"`;
  parametrize `replace`. Expect sample sizes `[0, 0, 3, 3]` in both modes.
  **Catches:** replacement bypassing the threshold (`[0, 3, 3, 3]`), and
  dropping an element with exactly `n_points` points.
- [x] **35. Invalid policy values.** `"maybe"` raises ValueError
  "allow_partial must be one of 'no', 'keep', 'drop'"; `1`, `None`, and
  `np.bool_(True)` raise TypeError "allow_partial must be a string".
  **Catches:** unknown strings silently accepted, and truthy values treated as
  the bool aliases.
- [x] **36. Indexed input uses its logical point count.** (optional, skipped) Select two
  of five source points through an indexed `PointCloudTensor`; with
  `n_points=3` and `allow_partial="drop"`, the sample is empty. **Catches:** a
  threshold counted on the backing source.
- [x] **37. The uniform threshold precedes duplicate removal.** (optional, skipped)
  Three identical points `[1.0]`, `n_points=3`, `allow_partial="drop"`, and
  `discard_duplicates=True` return one point, `[[1.0]]`. **Catches:** the check
  moved after duplicate removal. Item 32 covers the relative sampler.

## Streams and reporting

- [x] **38. Reseeding replays and calls advance.** Reference coordinates 0
  through 99, query `[0]`, `Uniform(0, 100)` (every point equally likely),
  `n_points=5`. One test: after `sb.random.seed(7)`, two calls with
  `generator` omitted differ from each other and give the same indices as the
  same two calls after a second `sb.random.seed(7)`. The global and caller
  generators share one advancing path, so a separate caller-generator test
  adds nothing. **Catches:** an omitted generator not reaching the global
  generator, a generator that does not advance, and nondeterministic seeding.
- [x] **39. Every output cell has its own stream.** The item 38 reference with
  `query=[0, 0]`, two equal `Uniform(0, 100)` distributions, `n_points=5`,
  `n_samples=2`, and `generator=sb.random.Generator(7)`: all eight cells
  differ. With a fixed seed the outcome is fixed; if the streams change, two
  of the 28 cell pairs match by chance with probability about 3e-9 (say so in
  a comment). **Catches:** streams keyed by query or distribution value
  instead of position, one stream block shared by all distributions, and every
  sample using the same engine.
- [x] **40. Results do not depend on the worker count.** (skipped: worker-count
  independence belongs to random generation, not subsampling) With equal seeds,
  `sb.system.limit_cpus(1)` and `limit_cpus(4)` give identical indices.
  Parametrize the number of queries over 1 and 7, covering nested sample
  parallelism and more queries than workers. Use the distributions
  `[Gaussian(0, 5), Gaussian(0, 2)]` on the item 38 reference with
  `n_points=5` and `n_samples=20` without replacement. Restore
  `limit_cpus(os.cpu_count())` in a fixture; there is no getter.
  **Catches:** scratch shared between partitions, and streams assigned in
  scheduling order. Only the without-replacement path uses per-partition key
  buffers, and both modes share the stream assignment.
- [x] **41. Seeding contract of `parallel_for_each_index_async`.** [C++]
  (optional) In `test/test_walk_random.cpp`, modeled on
  `ParallelWalkMatchesSequentialWalk`: index `i` receives
  `gen.reserve(count).sub_generator(i)`, and the generator advances by exactly
  `count`. **Catches:** a block that skips or repeats stream indices, or
  advances the generator by the wrong count. Item 40 covers this indirectly.
- [x] **42. Verbose mode names empty cells.** The item 31 example without
  replacement and with `verbose=True` emits the UserWarning "Empty samples at
  (query, distribution) indices: [(1, 0)]"; without `verbose`, there is no
  warning. Two small tests; do not assert progress-bar output. Background,
  not a third test: a cell with some but too few eligible points warns only
  under `"drop"`, which is why the example uses it; under `"keep"` it returns
  a partial sample silently. **Catches:** wrong or missing pairs, and warnings
  in quiet mode.
- [x] **43. Verbosity changes neither results nor generator state.** (optional, skipped)
  Two equally seeded generators, one used with `verbose=True` and one without,
  two calls each with the item 39 call, which has no empty cell and so emits no
  warning: all indices match. **Catches:** the reporting path consuming random
  numbers or reserving extra stream slots. Optional because `verbose` only
  polls progress counters and reads the empty-cell flags after the task
  completes.

## Reference representations

- [x] **44. An indexed reference.** Index a one-element `PointCloudTensor`
  holding coordinates 0, 4, 9, 20 with `sb.NestedTensor([sb.indices([3, 1])])`;
  its element `[0]` is a reference with logical points 20 and 4. Queries
  `[0, 1]` with `Uniform(0, 0.5)` and `n_points=1` select 20 and 4, with
  indices `[0]` and `[1]` relative to the view. **Catches:** reading the
  backing source instead of the view (points 0 and 4, or source indices 3
  and 1).
- [ ] **45. A `FloatTensor` coordinate query.**
  `sb.FloatTensor(np.array([[8], [3]], dtype=...))`, parametrized over float32
  and float64, against reference coordinates 0, 4, 9 with `Uniform(0, 1.5)`
  and `n_points=1` selects 9, then 4 (indices `[2]`, then `[1]`). **Catches:**
  a float32 query failing to convert to the float64 reference, and a float64
  `FloatTensor` passed on unconverted.
- [ ] **46. Samples ignore later writes to the reference.** Reference
  coordinates 0, 4, 9; query `[0]`; `Uniform(0, 0.5)`; `n_points=1`. After
  sampling, set `reference[0, 0] = 99.0`, confirm that the write took effect,
  and check that the sample still holds 0. **Catches:** samples sharing storage
  with the caller's reference instead of the task's copy. Uniform `subsample`
  has the equivalent test.

## Sampler input validation

- [ ] **47. The distribution argument.** (optional) `distribution=[]` raises
  ValueError "distribution list must not be empty"; a string raises TypeError
  "distribution must be a Distribution or a list of them". **Catches:** less
  helpful errors: without the Python checks, `[]` raises the native ValueError
  "expected one distribution or a nonempty distribution list", and a string
  raises AttributeError. Optional because only the error type and message are
  at stake.
- [ ] **48. Out-of-range indices.** On a three-point reference, `[3]`, `[-4]`,
  and `np.array([2**64 - 1], dtype=np.uint64)` raise ValueError "query index
  is out of range". **Catches:** unsigned values wrapping, and negative indices
  counting past the start. Pass the dtype explicitly:
  `np.asarray([2**64 - 1])` also wraps although its dtype prints as uint64, and
  so does the Python list `[2**64 - 1]`; both currently select the last point
  (see the follow-ups).
- [ ] **49. Index rank and type.** One parametrized test. A rank-2 `IntTensor`
  raises ValueError "query indices must be one-dimensional"; a 1-D float list
  raises TypeError "query indices must be integers"; a 3-D array raises
  ValueError "query must be a 2-D coordinate array or 1-D index vector". A 2-D
  integer array is a coordinate array by design, so use an `IntTensor` for the
  rank-2 case. **Catches:** a rank-2 index tensor accepted without the native
  one-dimensional check, and less helpful errors for float lists and
  higher-rank input, which still fail without their Python checks.
- [ ] **50. A coordinate query of the wrong dimension.** A 2-D coordinate
  query for a 1-D reference raises ValueError "reference and query must have
  the same dimension". **Catches:** a dimension mismatch reaching the distance
  computation, where it would read past the reference coordinates (only this
  native check guards it). A coordinate query on a distance-matrix reference
  needs no test: without the Python check, the native code still raises.
- [ ] **51. Unsupported query and reference objects.** (optional) A
  `PointCloudTensor` as the query raises TypeError "query must be one point
  cloud, a coordinate array, or an index vector", and as the reference
  TypeError "reference must be one PointCloud or DistanceMatrix". **Catches:**
  tensors reaching the conversion code with a misleading error. Optional
  because both are one-line type guards.
- [ ] **52. Nonfinite coordinates.** (optional) An infinite reference
  coordinate and a NaN coordinate query each raise ValueError "coordinates must
  be finite". **Catches:** a less helpful error: without the check, a nonfinite
  coordinate raises OverflowError "Euclidean distance exceeds numerical range"
  instead. Optional because only the error type and message are at stake.
- [ ] **53. NaN written into a matrix.** (optional) Entry assignment rejects
  only negative values, so `matrix[0, 1] = nan` succeeds, and sampling then
  raises ValueError "distances must be nonnegative and not NaN". **Catches:**
  the reference check removed, leaving only the less specific "values must not
  be NaN" from the weight evaluation. `test_distance_matrix.py` already covers
  constructor rejection of NaN and negative values.
- [ ] **54. Generator argument type.** Passing a NumPy generator,
  `np.random.default_rng(0)`, as `generator` raises TypeError "generator must
  be a stablebear.random.Generator or None". **Catches:** the sampler bypassing
  the shared `_unwrap`, so the object reaches pybind11 with an opaque error.
  `subsample`, `noisy_sin`, `noisy_cos`, and `sample_poisson` call the same
  helper; one caller pins the message.

## Distribution API

These go in `test/python/test_distributions.py`.

- [ ] **55. Printing.** Two small tests. `str`: the docstring example
  `str(Mixture([Gaussian(0, 0.5), Uniform(0, 2.5)], [1, 3]))` is
  "0.25 * Gaussian(0, 0.5) + 0.75 * Uniform(0, 2.5)", and the nested mixture
  `Mixture([Mixture([Uniform(0, 1), Uniform(1, 2)], [1, 1]), Gaussian(0, 1)], [1, 0])`
  prints "1 * (0.5 * Uniform(0, 1) + 0.5 * Uniform(1, 2)) + 0 * Gaussian(0, 1)".
  `repr`: the first mixture's repr is
  "Mixture(distributions=(Gaussian(mean=0.0, std=0.5), Uniform(start=0.0, end=2.5)), coefficients=(0.25, 0.75))";
  checking `eval(repr(m)) == m` is optional. **Catches:** raw coefficients,
  nested mixtures without parentheses, and dropped zero terms. The nested zero
  term is also the only check that mixtures keep zero-coefficient components,
  as the docstring promises.
- [ ] **56. Distributions are immutable.** Assigning to `.mean` of
  `Gaussian(2, 0.5)` raises AttributeError "Gaussian is immutable". Match the
  message: without the guard, Python raises a different AttributeError for the
  read-only property. **Catches:** mutable distributions whose hash could
  change.
- [ ] **57. Return types of `.weight()`.** Two small tests.
  `Gaussian(0, 1).weight(0)` and a 0-d array each return an object whose type is
  exactly `float` (`type(result) is float`; `isinstance` passes for
  `numpy.float64`, which subclasses `float`), equal to 0.3989422804. The
  float32 array `[[0, 2], [3, 1.5]]` gives `Uniform(0, 2)` weights
  `[[0.5, 0], [0, 0.5]]` as float64. **Catches:** returning a 0-d array or a
  NumPy scalar for scalar input, flattened output, and a float32 result.
- [ ] **58. Evaluation validation.** Two small tests. NaN and `[0, -inf]`
  raise ValueError "values must be finite"; `1 + 2j` and the numeric string
  `"1"` raise TypeError "values must be real numbers". **Catches:** nonfinite
  values producing silent zeros (`[0, -inf]` checks that every element is
  examined), and complex values or numeric strings silently read as real
  numbers: without the dtype check, both evaluate as 1. Item 2 covers finite
  negative values. Booleans are
  currently accepted; leave them out until that is decided (see the
  follow-ups).
- [ ] **59. Gaussian parameters.** Two small tests. `std` 0, -1, or inf, or a
  NaN `mean`, raises ValueError "Gaussian requires a finite mean and finite
  positive std"; `mean=True` or `"0"` raises TypeError "mean must be a real
  scalar" (the shared `_real`, checked once here). **Catches:** nonpositive or
  infinite scales producing NaN or zero weights, and bools or strings accepted
  as numbers.
- [ ] **60. Uniform parameters.** One test. A NaN endpoint, `(1, 1)`, and
  `(2, 1)` raise ValueError "Uniform requires start < end (start may be -inf
  and end may be +inf)". Infinite endpoints with `start < end` are valid;
  item 6 constructs them. **Catches:** empty, reversed, or NaN intervals
  producing zero or NaN weights.
- [ ] **61. Mixture inputs.** One table of ValueErrors: one distribution with
  two coefficients ("Mixture needs one coefficient per distribution"), a
  negative or NaN coefficient ("Mixture coefficients must be finite and
  nonnegative"), and all-zero coefficients ("Mixture needs at least one
  positive coefficient"). **Catches:** silently truncated pairs and
  coefficients that cannot be normalized. An empty mixture and a
  non-Distribution component need no rows: they still fail without their
  checks.

## Plot ranges and heatmap

Items 63 and 64 go in `test/python/test_distributions.py`; items 65-71 go in
`test/python/test_plotting.py`.

- [ ] **62. Confirm the double margin.** (decision) Uniform's plot range adds
  10% of its width, and the heatmap's automatic extent adds another 10% of the
  largest absolute bound (both documented). Decide whether both margins are
  intended. If either changes, change the code first and adjust the expected
  values in items 63, 64, and 66 before writing them. No test.
- [ ] **63. Single-distribution plot ranges.** One parametrized test with
  `pytest.approx`: `Gaussian(2, 0.5)` gives `(0.5, 3.5)` and `Uniform(0, 2)`
  gives `(-0.2, 2.2)`. An infinite endpoint is first replaced to give a
  unit-width interval, then widened by 10%: `Uniform(1, inf)` gives
  `(0.9, 2.1)`, `Uniform(-inf, 0)` gives `(-1.1, 0.1)`, and
  `Uniform(-inf, inf)` gives `(-0.1, 1.1)`. **Catches:** infinities leaking
  into the range, the margin applied before the replacement, and the unit
  window on the wrong side.
- [ ] **64. Mixture plot ranges span positive-coefficient components.**
  `Mixture([Mixture([Uniform(0, 1), Uniform(10, 20)], [1, 0]), Uniform(2, 3)], [1, 1])`
  gives `(-0.1, 3.1)`; including the zero-coefficient component would give
  `(-0.1, 21.0)`. **Catches:** zero-weight components extending the range, and
  nested mixtures not recursing.
- [ ] **65. Heatmap geometry and values.** `Uniform(0, 1.1)`, query `(1, 1)`,
  extent `(-0.5, 2.5, 0.5, 3.5)`, resolution 3. Pixel centers are x = 0, 1, 2
  and y = 1, 2, 3, so the image rows are `[10/11, 10/11, 10/11]`,
  `[0, 10/11, 0]`, and `[0, 0, 0]`. Also assert that `im.get_extent()` equals
  the given extent and that `im.origin` is `"lower"`, which puts the first row
  at the bottom. Compare values with `pytest.approx`. **Catches:** pixel edges
  instead of centers, axiswise (Chebyshev) instead of Euclidean distance, an
  ignored query offset, transposed rows, a changed origin, and normalization
  over the grid (1/4 per nonzero pixel).
- [ ] **66. Automatic extent.** One parametrized test with `pytest.approx`:
  `Uniform(0, 2)` with query `(10, -5)` gives `(7.58, 12.42, -7.42, -2.58)`,
  and `Uniform(0, inf)` with query `(0, 0)` gives
  `(-1.21, 1.21, -1.21, 1.21)`, so unbounded support needs no explicit extent.
  The half-width is 1.1 times the largest absolute `plot_range()` bound.
  **Catches:** an extent not centered on the query, a missing margin, and
  unbounded ranges failing.
- [ ] **67. Documented defaults.** (optional) With `extent` and `resolution`
  omitted, the image is 512 by 512, `ax.get_aspect() == 1.0`, the colormap is
  viridis, and interpolation is nearest. **Catches:** changed defaults.
  Optional because these are Matplotlib keyword defaults listed in the
  docstring; item 65 checks the origin.
- [ ] **68. Nonfinite plot range.** (optional) `Gaussian(0, 1e308)` without
  an extent raises ValueError "cannot choose a finite plot region; pass extent
  explicitly", and works with `extent=(-1, 1, -1, 1)`. Any range whose largest
  absolute bound times 1.1 overflows, such as `Uniform(0, 1.7e308)`, reaches
  the same error. **Catches:** an overflowing radius reported only as the
  generic extent error, without telling the caller to pass an extent.
- [ ] **69. Supplied and current axes.** Two small tests. With `ax` given while
  another axes is current, the image lands on the given axes, its existing line
  remains, and the current axes gets no image. With `ax` omitted, the image
  goes to the current axes. The `ax` fixture from `plot_helpers.ax_fixture` is
  itself current, so the second test creates and selects another axes
  explicitly. **Catches:** drawing on `plt.gca()` regardless of `ax`, and
  clearing the given axes.
- [ ] **70. Styling keywords override defaults.** `cmap="magma"`,
  `interpolation="bilinear"`, and `alpha=0.5` reach the returned image.
  **Catches:** defaults overwriting caller keywords. Do not test colorbars
  (Matplotlib's behavior) or `origin=` (see the follow-ups).
- [ ] **71. Heatmap argument validation.** Two small tests. Queries `(0,)`,
  `(0, nan)`, and `(1j, 0)` raise ValueError "query must contain two finite
  real coordinates", and a non-Distribution raises TypeError "distribution must
  be a Distribution". The extent `(0, 1, 0, inf)` raises ValueError "extent
  must contain four finite real bounds", and `(0, 0, 0, 1)` and `(0, 1, 1, 0)`
  raise "extent must satisfy xmin < xmax and ymin < ymax"; the query and the
  extent share one length check, which `(0,)` covers.
  Resolution uses the shared `_positive_integer`, already tested in
  `test_subsample.py`. **Catches:** malformed queries reaching the grid
  computation, and reversed or degenerate extents producing flipped or empty
  images.

## Composition and final checks

- [ ] **72. Persistence pipeline.** (optional) One test parametrized over a
  point cloud and its pairwise distance matrix: points (0, 0), (1, 0), (0, 1),
  (10, 0), and (11, 0); queries `[0, 3]`; `Uniform(0, 1.5)`; `n_points=3`;
  `n_samples=2`. Every eligible point is drawn, so draw order does not matter.
  Persistent homology with `max_dim=0`
  (`persistence.compute_persistent_homology`), then
  `persistence.barcode_to_stable_rank`, then `sb.mean(..., dim=1)` gives shape
  `(2, 1)`. The averaged stable rank is 3 on `[0, 1)` and 1 afterward for the
  first query, and 2 then 1 for the second. **Catches:** axes lost or reordered
  between sampling and the reduction. Optional because items 11 and 12 cover
  the sampler's axes, and `persistence/test_ripser.py` already runs persistent
  homology on indexed point-cloud and distance-matrix tensors.
- [ ] **73. Final test run.** Build and install with CMake using
  `-j$(nproc --ignore=4)`, as in AGENTS.md. From `test/`, run the new tests and
  the uniform-sampling suite on the default backend and with `SB_FORCE_CPU=1`,
  and run `sb_test` for the C++ items. Record the checks actually run.
- [ ] **74. Documentation check.** Build the documentation, run the
  `docs/subsampling_options.rst` snippets by hand (no test runner executes
  them), regenerate the heatmap figures with
  `python docs/_static/gen_weight_function_fig.py`, and inspect one heatmap
  for smoothness and useful extents. Record the checks actually run.

## Coverage map: log-space sampling code

| Code path | Regression | Items |
| --- | --- | --- |
| Gaussian log density, shared by `weight()` and `log_weight()` | wrong exponent, mean, scale, or constant | 1 (all), 15 (exponent sign), 16 (mean and std), 19 (constants inside a mixture) |
| Uniform `log_weight` as `log(weight)` | wrong endpoints or density | 2, 6, 17, 19, 20, 27 |
| `Mixture::log_weight`, used by sampling | wrong coefficient logs, `log_add`, nesting, or redistribution | 17, 18, 19, 20, 21, 25 |
| `Mixture::weight`, used by `.weight()` and heatmaps | wrong sum or redistribution | 5 |
| Infinite distances mapped to `-inf` before evaluation | Gaussian raising on infinite matrix distances | 27 |
| `normalize_log_row` shifting by the row maximum | a far query losing every eligible point | 24 |
| With replacement: CDF from probabilities, inverse-CDF draws | wrong law, zero-weight points drawn, endpoint rounding | 15, 22, 23 |
| Without replacement: log exponential keys, difference comparator | wrong first or second draw, equal tails ordered by index | 20, 21, 24, 26 |
| Compaction and mapping back to source indices | compact positions returned; weights out of line with their source indices | 7, 20, 21 |
| Eligibility as a finite log weight | tiny weights dropped, policies counting the wrong points | 8, 25-31 |
| Euclidean distance between coordinates | another metric | 14 |
| Per-query generators and per-distribution stream blocks | shared or reordered streams | 38-40 |

The two mixture paths are separate implementations: tests of one do not
protect the other.

## Deliberately not covered

- Exact seeded outputs (goldens), for the reasons under the conventions.
- Progress-bar text, the progress total, and timing.
- Cancellation, which is checked only before each query, distribution row,
  and sample. Check it manually if that code changes.
- Overflow branches reachable only where `long double` is `double`: Euclidean
  distance overflow, the Gaussian log-weight overflow error, and the Uniform
  huge-width fallback. x86-64 Linux cannot reach them.
- Details the public API cannot observe: subtracting `log(sum)` in
  `normalize_log_row` (draws scale to the CDF total, and keys compare
  differences), the `count != 0` guard in `draw_samples`, and skipping zero
  coefficients in `Mixture::log_weight` (an evaluated zero coefficient adds
  `log(0) = -inf`).
- Negative distances at the sampler: `DistanceMatrix` rejects them on
  construction and assignment.
- Statistical checks on distance matrices or float32: weight rows are shared
  code, and item 13 covers those paths deterministically.
- A Mixture inside a distribution list counting as one entry: only a list
  adds an axis.
- Undocumented stream layout, such as query results not depending on later
  queries.
- Pixel snapshots of heatmaps; item 74 inspects one figure by eye.
- Indexed storage and serialization of relative output: it uses the same
  indexed storage types as uniform `subsample`, which
  `test_indexed_tensor_io.py` round-trips, and a save/load and pickle round
  trip of all four kinds was confirmed by hand. Keep mutation, copy-on-write,
  and serialization tests in those owning files.

## Follow-ups outside this checklist

- **Release notes: `replace` now defaults to True.** Both `subsample` and
  `subsample_relative` changed their default from `replace=False`. For
  `subsample`, released in 0.5.0, this is a breaking change; list it under
  breaking changes in the next release's CHANGELOG entry.
- **`IntTensor` wraps Python integers of `2**63` and above** (predates the
  branch). `np.asarray([2**64 - 1])` has scalar type `numpy.ulonglong`, which
  `IntTensor` does not map to uint64 (the `IntTensor.__init__` lookup in
  `stablebear/base_tensor.py`, using `_NP_TO_SB` in `stablebear/typing.py`),
  so it falls back to int64 and wraps:
  `subsample_relative(reference, [2**64 - 1], ...)` silently selects the last
  point. Fix it with its own test in `test_int_tensor.py`, then add the list
  case to item 48.
- **A failed `"no"` relative call advances the generator.** Seeds are reserved
  before the workers find the shortfall, whereas uniform `subsample` validates
  first and has a test that it does not advance. Decide whether to document
  this as unspecified or change it; add no test until then.
- **Adjacent seeds share streams** (predates the branch). A call's stream
  block starts at `seed + offset`, so `Generator(5)` and `Generator(6)`
  produce shifted copies of the same streams. Consider a seed mixer
  separately.
- **Equal mixtures that sample differently.** A normalized coefficient can be
  positive in `long double` yet round to 0.0 in double:
  `Mixture([Uniform(0, 1), Uniform(5, 6)], [1e308, 1e-16])` equals and hashes
  like the `[1, 0]` mixture, but sampling still uses its second component.
  Consider flushing such coefficients to zero at construction.
- **Booleans in `.weight()`.** `Gaussian().weight(True)` returns the weight at
  1, although `_real` rejects bool parameters, and `Gaussian().weight(2**70)`
  raises the misleading TypeError "values must be real numbers". Decide before
  extending item 58.
- **`plot_range()` is not always finite or nondegenerate.**
  `Gaussian(0, 1e308)` gives `(-inf, inf)`, `Uniform(0, 1.7e308)` gives
  `(-1.7e307, inf)`, and `Uniform(1e17, inf)`, `Uniform(-inf, 1e308)`, and
  `Gaussian(1.7e308, 1)` give zero-width ranges, although the docstring
  promises finite bounds.
- **Tiny scales.** `.weight()` returns inf for valid tiny scales such as
  `Uniform(0, 5e-324).weight(0)`, because a finite `long double` is cast to
  double. Sampling is unaffected.
- **`DistanceMatrix` entry assignment accepts NaN**, unlike the constructor.
  If that changes, drop item 53.
- **Uniform `subsample` verbosity** is feature work, not a test: offer opt-in
  `verbose=True` with progress and empty-sample diagnostics naming the input
  dataset, documented under the shared subsampling options. Track it as an
  issue and test it when implemented.
- **Low priority.** `del dist._native` is not blocked; a custom `Distribution`
  subclass passes the isinstance checks and then fails with an opaque pybind11
  error; passing `origin=` to the heatmap raises Python's duplicate-keyword
  TypeError from the `Axes.imshow` call; the without-replacement comparator
  could lose strict weak ordering for keys within a few ulps (statistically
  negligible); and `Generator.reserve` is unsynchronized, so Python threads
  sharing one generator can race (predates the branch).
- **Docs and comments.** `CHANGELOG.md` still says unbounded heatmap ranges
  require an explicit extent and calls `.weight()` inputs "filter values". The
  `subsample_relative` docstring defines eligibility as positive weight, but
  eligibility is a finite log weight, and `.weight()` can print 0 for eligible
  far-tail points. The `log_probabilities_to_cdf` comment in
  `weighted_draw.hpp` says the normalized row has "maximum 0"; its maximum is
  `-log(sum)`, at most 0.
