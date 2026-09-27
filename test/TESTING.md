# Testing manifesto

Tests should be simple, transparent, and proportionate to the behavior they
protect. These principles apply when writing, reviewing, or changing tests.

1. **A test should be understandable in one reading.** Its name, input,
   action, and expected outcome should tell the same story.

2. **Test one behavior at a time.** Multiple assertions are fine when they
   establish that behavior. Independent guarantees belong in separate tests.

3. **Make expectations obvious.** Prefer small, explicit inputs and expected
   values. Avoid clever expressions, unexplained indices, and reference
   implementations that are algorithms themselves.

4. **Test the public contract.** Use public APIs unless the behavior genuinely
   cannot be observed there. Keep internal-storage tests separate from
   functional tests.

5. **Keep tests in scope.** Sampling tests test sampling; tensor tests test
   views, mutation, copying, and casting.

6. **Prefer clarity over reuse.** Share a fixture when it makes the setup
   easier to understand. Split tests when sharing requires type-dependent
   branches or hides important differences. Some duplication is acceptable.

7. **Use representative, distinguishable data.** Different input elements
   should have values that expose mix-ups. Use degenerate inputs only when
   testing their specific behavior.

8. **Make mutation checks meaningful.** Verify the intended write succeeded
   before asserting that something else stayed unchanged. Name snapshots by
   what they preserve.

9. **Parametrize meaningful variations.** Cover supported precisions and
   relevant cases, but avoid unnecessary combinations. Every parameter should
   have a clear purpose.

10. **Preserve coverage, not test count.** Before adding a test, check for
    existing coverage. Move, extend, or consolidate it rather than repeat it.

11. **Use fixed goldens for format compatibility.** Don't construct expected
    files by splicing bytes from the current writer. Keep malformed-input
    tests focused and separate.

12. **Keep test infrastructure proportionate.** Avoid custom frameworks,
    executors, monkeypatching, or allocation instrumentation unless essential
    to the specific guarantee.

13. **Formatting matters.** Lay out test inputs and expected values so their
    structure is visible. Write matrices with one row per line rather than
    compressing them into `[[a, b], [c, d]]`:

    ```python
    [[a, b],
     [c, d]]
    ```

    For collections of matrices, use blank lines between matrices to make
    their boundaries clear.

14. **Write small test inputs explicitly.** Prefer listing a few entries over
    generating them with a loop or comprehension, even when they repeat. For
    example, four explicit `sb.indices([0, 1])` entries make the selections
    easier to see than `sb.indices([0, 1]) for _ in range(4)`. Use generated
    data when the size or pattern is itself relevant to the behavior under test.

15. **Use short comments where intent could be unclear.** A brief comment can
    explain what a selection picks out, why an input matters, what an
    operation is meant to demonstrate, or which guarantee an assertion checks.
    Add one when it helps the reader
    understand the test; avoid narrating code that is already obvious.

The review question is: **What bug would make this test fail, and can I see
that immediately?**

Build and test commands are documented in [AGENTS.md](../AGENTS.md#testing).
