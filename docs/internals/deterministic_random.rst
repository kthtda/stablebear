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

   m_seed + flatIndex
     -> make_engine<Xoroshiro128pp>
       -> s0 = splitmix64(seed)
       -> s1 = splitmix64(s0)
       -> Xoroshiro128pp(s0, s1)

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

``RandomGenerator<EngineT>`` stores a single ``uint64_t`` master seed. It holds no RNG state. To produce an engine for element ``i``, it computes ``make_engine<EngineT>(m_seed + i)``:

.. code-block:: cpp

   template <typename EngineT = Xoroshiro128pp>
   class RandomGenerator
   {
   public:
     explicit RandomGenerator(uint64_t seed);

     EngineT sub_generator(size_t flatIndex) const
     {
       return detail::make_engine<EngineT>(m_seed + flatIndex);
     }
   };

Since ``flatIndex`` is the row-major index into the tensor, it is a pure function of position -- not of execution order.


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

Internally, both variants compute the flat (row-major) index for each element, then call ``gen.sub_generator(flatIndex)`` to create a fresh engine for that element. The callback receives the engine by reference and can draw as many samples as needed.

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
