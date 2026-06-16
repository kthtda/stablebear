====================================
Deterministic random generation
====================================

stablebear's random generation system produces reproducible results regardless of thread count or execution order. This is achieved by deriving a unique, deterministic seed for each tensor element from a master seed and the element's position, so that parallel threads never share or depend on each other's random state.

This mechanism underpins all random operations in stablebear -- generating noisy PCFs, sampling Poisson point processes, and any future operation that needs per-element randomness.


Design goal
============

Several operations populate tensors element-wise with random data: ``sample_poisson`` fills a point cloud tensor with Poisson-distributed points, ``noisy_sin`` fills a PCF tensor with noisy discretizations, and so on. These operations are parallelized across threads via Taskflow :footcite:`Huang2021`.

In a parallel setting, the order in which elements are visited depends on the thread scheduler -- this is non-deterministic. A shared RNG would produce different results on every run. The requirement is: **same seed, same results, always** -- independent of the number of threads, the order of execution, or the platform.


Architecture
=============

The system has three layers:

1. **Engine** (``xoroshiro128pp.hpp``) -- the random engine that produces pseudorandom output from a given state.
2. **RandomGenerator** (``random_generator.hpp``) -- holds a master seed and derives per-element engines via ``make_engine``.
3. **Walk with random** (``walk.hpp``) -- tensor traversal that pairs each element with its own deterministically-seeded engine.

Any function that needs per-element randomness simply calls ``walk()`` or ``parallel_walk()`` with a ``RandomGenerator``, receiving an independent engine at each element. The function does not need to know anything about the seeding strategy.


Engine: xoroshiro128++
=======================

The default engine is ``Xoroshiro128pp``, an implementation of xoroshiro128++ :footcite:`Blackman2021`. It satisfies the C++ ``UniformRandomBitGenerator`` concept, so it works with all standard distribution types (``std::normal_distribution``, ``std::poisson_distribution``, etc.).

The engine takes two 64-bit state words (``s0``, ``s1``) and produces 64-bit output. The core algorithm lives in ``detail/xoroshiro128pp_impl.hpp`` (public domain reference code, unmodified), while the engine wrapper is in ``xoroshiro128pp.hpp``.

Key properties:

- **128-bit state** -- two ``uint64_t`` words, vs 2.5 KB for ``mt19937_64``
- **Period** -- :math:`2^{128} - 1`, far more than needed for per-element streams
- **Quality** -- passes BigCrush :footcite:`LEcuyer2007`
- **Speed** -- significantly faster than both ``mt19937_64`` and counter-based alternatives (Philox) for the init-then-draw pattern used here


Seeding via make_engine
========================

The ``RandomGenerator`` class is templated on the engine type. It does not construct engines directly -- instead, it delegates to ``detail::make_engine<EngineT>(seed)``, which encapsulates the engine-specific seeding strategy.

For the default engine, the seeding chain is:

.. code-block:: text

   m_seed + m_offset + flatIndex
     -> make_engine<Xoroshiro128pp>
       -> s0 = splitmix64(seed)
       -> s1 = splitmix64(s0)
       -> Xoroshiro128pp(s0, s1)

(``m_offset`` is the per-generator stream counter described in
:ref:`advancing-between-draws` below; it is ``0`` for the first draw after
seeding, so the chain reduces to ``m_seed + flatIndex`` there.)

This follows the recommended seeding approach from :footcite:`Blackman2021`: chain ``splitmix64`` outputs to fill the state words, feeding each output as the next input. Since ``splitmix64`` is a bijection, this guarantees two distinct, well-distributed state words and avoids the all-zeros state (which xoroshiro cannot be in).

The default ``make_engine`` for other engine types applies a single ``splitmix64`` and forwards to the engine's constructor:

.. code-block:: cpp

   template <typename EngineT>
   EngineT make_engine(uint64_t seed)
   {
     return EngineT(splitmix64(seed));
   }

Adding support for a new engine type requires only a ``make_engine`` specialization -- the rest of the system (``RandomGenerator``, walk, consumer functions) is unaffected.


splitmix64
-----------

Raw addition (``seed + index``) would produce correlated seeds for nearby indices. The ``splitmix64`` hash function :footcite:`Steele2014` transforms the sum into a well-distributed 64-bit value:

.. code-block:: cpp

   uint64_t splitmix64(uint64_t x)
   {
     x += 0x9e3779b97f4a7c15ULL;
     x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
     x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
     return x ^ (x >> 31);
   }

This ensures that adjacent indices produce statistically independent seed values.


RandomGenerator
================

``RandomGenerator<EngineT>`` stores a ``uint64_t`` master seed and a ``uint64_t`` offset -- a counter of how many seed slots have been handed out so far. It holds no RNG *engine* state. A draw does not pull engines from the generator directly; instead it **reserves a contiguous block** of seed slots via ``reserve(n)`` and derives one engine per element from that block:

.. code-block:: cpp

   template <typename EngineT = Xoroshiro128pp>
   class RandomGenerator
   {
   public:
     explicit RandomGenerator(uint64_t seed);

     // A reserved, contiguous range of seed slots. Captured by value, so it
     // stays valid even after the generator advances or the draw runs async.
     class Block
     {
     public:
       EngineT sub_generator(size_t flatIndex) const
       {
         return detail::make_engine<EngineT>(m_base + flatIndex);
       }
     private:
       uint64_t m_base;
     };

     // Reserve the next n slots [m_offset, m_offset + n) and advance past them.
     Block reserve(size_t n)
     {
       Block block(m_seed + m_offset);
       m_offset += n;
       return block;
     }
   };

Within a single draw, element ``i`` is seeded from ``m_base + i`` where ``m_base = m_seed + m_offset``. Since ``flatIndex`` is the row-major index into the tensor, the engine for each element is a pure function of position -- not of execution order. Across draws, the offset moves forward (see below), so successive draws occupy disjoint ranges of seed slots.


.. _advancing-between-draws:

Advancing between draws
=======================

A generator must advance its state between sampling calls. Otherwise two consecutive draws from the same generator would reuse the same seed slots and produce **byte-for-byte identical** tensors -- a silent footgun for any repeated-sampling loop (bootstrap, Monte Carlo), and contrary to the ``numpy.random.Generator`` / ``torch.Generator`` convention:

.. code-block:: python

   g = sb.random.Generator(seed=5)
   a = sb.random.noisy_sin((3,), generator=g)
   b = sb.random.noisy_sin((3,), generator=g)
   # a and b differ; a fresh Generator(5) reproduces both, in order.

Advancing is the job of ``reserve(n)``: a draw over an ``N``-element tensor reserves the block ``[m_offset, m_offset + N)`` and bumps ``m_offset`` by ``N`` **immediately**, before any element is filled. The next draw therefore starts at ``m_offset = N`` and cannot overlap the previous one. Re-seeding (``seed()``) resets the offset to ``0``.

Reserving up front -- rather than advancing *after* the draw -- is what makes this correct under parallelism. ``parallel_walk`` dispatches its per-element callback asynchronously via Taskflow, so the callbacks run *after* the call returns. If the generator were advanced only once the draw "finished", there would be no well-defined moment to do so, and reading the live generator's offset from inside the async callbacks would race. Instead, ``reserve`` computes the block base and advances the counter synchronously, then the immutable ``Block`` (a single ``uint64_t``) is captured **by value** into the callback. Workers read only their copy; the generator is free to advance for the next draw.

**Backward compatibility.** The first draw after seeding uses ``m_offset == 0``, so its element ``i`` is seeded from ``m_seed + i`` -- exactly the pre-advancement behaviour. Existing seeds reproduce their historical first draw unchanged; only the *second and later* draws (previously duplicates) now differ.

**Non-overlap by construction.** Because each draw reserves a disjoint, contiguous slot range, independence between draws is structural -- it does not rely on a hash scattering nearby seeds apart. (``splitmix64`` still decorrelates adjacent slots *within* the engine seeding chain, as described above.)


Walk integration
=================

The ``walk()`` and ``parallel_walk()`` functions in ``walk.hpp`` provide the bridge between the random generator and tensor traversal. Both sequential and parallel variants exist:

.. code-block:: cpp

   // Sequential: visits every element in row-major order
   walk(tensor, generator, [](const std::vector<size_t>& idx, EngineT& engine) {
     // engine is seeded deterministically from idx's flat position
   });

   // Parallel: distributes elements across threads via taskflow
   parallel_walk(tensor, generator, [](const std::vector<size_t>& idx, EngineT& engine) {
     // same engine for same idx, regardless of which thread runs this
   }, executor);

Internally, both variants first call ``gen.reserve(tensor.size())`` once to claim a block for the whole draw (advancing the generator), then compute the flat (row-major) index for each element and call ``block.sub_generator(flatIndex)`` to create that element's engine. The ``Block`` is captured by value, so the parallel variant is safe even though its callbacks run asynchronously. The callback receives the engine by reference and can draw as many samples as needed.

This is the only entry point for deterministic randomness. A consumer function does not manage seeds or engines directly -- it receives a ready-to-use engine from the walk and draws from it:

.. code-block:: cpp

   // Example: sample_poisson uses parallel_walk with a generator
   sb::parallel_walk(out, gen,
     [&](const std::vector<size_t>& idx, auto& engine) {
       std::poisson_distribution<size_t> countDist(lambda);
       auto nPoints = countDist(engine);
       // ... fill point cloud using engine ...
     }, exec);

Adding a new random operation follows the same pattern: accept a ``RandomGenerator``, pass it to ``walk`` or ``parallel_walk``, and use the provided engine.


Why flat indexing works
------------------------

The flat index is computed from the multi-dimensional index and the tensor shape:

.. code-block:: text

   For a tensor with shape (s0, s1, ..., sN):
     flat(i0, i1, ..., iN) = i0 * (s1 * s2 * ... * sN) + i1 * (s2 * ... * sN) + ... + iN

This mapping is bijective -- every element has a unique flat index, and the index depends only on position and shape, not on traversal order. The parallel walk computes the same flat index from any thread:

.. code-block:: cpp

   // In parallel_walk_impl:
   size_t rem = flat;
   for (ptrdiff_t i = ndim - 1; i >= 0; --i)
   {
     idx[i] = rem % shape[i];
     rem /= shape[i];
   }

This reverse computation reconstructs the multi-index from the flat index, ensuring both sequential and parallel walks agree on the mapping.


Concrete example
=================

Consider a ``(3, 4)`` tensor populated with seed 42:

.. code-block:: text

   Element (0, 0): flat = 0  -> make_engine(42 + 0)  -> splitmix64 chain -> engine
   Element (0, 1): flat = 1  -> make_engine(42 + 1)  -> splitmix64 chain -> engine
   Element (0, 2): flat = 2  -> make_engine(42 + 2)  -> splitmix64 chain -> engine
   ...
   Element (2, 3): flat = 11 -> make_engine(42 + 11) -> splitmix64 chain -> engine

Each element gets its own ``Xoroshiro128pp`` engine with a unique state derived from its position. Whether the walk visits these 12 elements on 1 thread or 12 threads, each element always receives the same engine, producing identical results.

The draw reserves slots ``[0, 12)`` and advances the offset to ``12``, so a **second** draw from the same generator shifts every base by 12:

.. code-block:: text

   Second draw, element (0, 0): flat = 0  -> make_engine(42 + 12 + 0)  -> engine
   Second draw, element (2, 3): flat = 11 -> make_engine(42 + 12 + 11) -> engine

The two draws never share a seed slot, so they differ -- yet replaying with a fresh ``Generator(42)`` reproduces both draws in the same order.


Global and explicit generators
===============================

The system supports two usage patterns:

- **Global generator** -- ``sb::seed(42)`` sets the seed on a process-wide ``DefaultRandomGenerator`` (``default_generator()``). Some functions use this by default when no generator is passed explicitly.
- **Explicit generator** -- ``RandomGenerator gen(42)`` creates an independent generator that can be passed to functions, allowing multiple independent random streams.


Switching engines
==================

The engine is a template parameter on ``RandomGenerator``. To use a different engine:

1. Implement the engine class satisfying ``UniformRandomBitGenerator`` (``result_type``, ``operator()``, ``min()``, ``max()``).
2. Add a ``detail::make_engine`` specialization if the engine needs a non-standard seeding strategy.
3. Instantiate ``RandomGenerator<YourEngine>``.

The walk infrastructure, consumer functions, and Python bindings do not change.


References
==========


.. footbibliography::
