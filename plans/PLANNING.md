# Planning manifesto

An implementation plan turns a spec into a sequence of small changes that can
be made and reviewed one at a time. These principles apply when writing,
reviewing, or following a plan in this folder.

A spec says what to build and why. A plan says in which order to build it.
Keep them in separate files, and link the plan to its spec.

## Structure

1. **Group items into phases.** A phase is a milestone: when its last item is
   done, something coherent works, such as a new property with its tests and
   docs. Name each phase by what it delivers, not by the kind of work in it.

2. **Letter the phases and number the items within them.** Phases are A, B,
   C, ..., and items are A1, A2, ..., then B1, B2, ..., so that an item can be
   referred to by its label alone. Small steps within an item are subitems,
   A1a, A1b, ..., listed under it. Do not relabel items once work has
   started; add a new item at the end of its phase, or, if it belongs with an
   existing item, as a subitem of that item.

3. **Order items so they can be done top to bottom.** Each item may depend
   only on items above it. If two items can be done in either order, list
   them in the order that is easier to review.

4. **Use a checkbox per item and subitem.** Write `- [ ] **B2.** ...`, and
   indent subitems under their item as `- [ ] **B2a.** ...`. Check an item
   once all its subitems are checked.

5. **Check an item in the commit after it, with its hash.** Once an item's
   commit exists, check the item and add the commit's short hash as
   evidence, as in `- [x] **B2b.** [3] (9629137bd, d37757337) ...`. The tick and
   hash go into the next item's commit. A commit cannot record its own
   hash, because amending it to add the hash changes the hash.

## Item size

6. **One item is one reviewable change.** It should fit in one commit that a
   reviewer can read in one sitting. If a description needs "and" to join
   unrelated changes, split it.

7. **Each item leaves the build and the tests passing.** No item should
   depend on a later one to compile or pass. Introduce a function before its
   callers, and a type before its users.

8. **Say what is done, not just what to work on.** "Add `Times.epoch`,
   stored in C++, read-only in Python" is an item; "Work on epochs" is not.
   Name the files or functions involved when they are known.

## Tests and docs

9. **Write tests with the code they cover.** Tests belong in the same item
   as the behavior they check, or in the item right after it. Do not
   collect them into a testing phase at the end.

10. **Name the tests in each item.** List the few cases an item adds, such
    as "tests: default is `None`; `to` keeps it; `__eq__` compares it". An
    item should add a handful of tests, not "write the tests for X". Follow
    the [testing manifesto](../test/TESTING.md).

11. **Update docs in the item that changes behavior.** A public API change
    and its docstring belong together. Larger guide pages may be their own
    item at the end of a phase. Follow the
    [writing manifesto](../docs/WRITING.md).

## Keeping the plan current

12. **Change the plan when the work changes.** If an item turns out to be
    too large, split it before starting. If a decision changes the design,
    update the spec first, then the plan.

13. **Record what was learned, briefly.** A short note under an item, such
    as a surprise found while doing it, helps the next items. Do not turn the
    plan into a log.

14. **Estimate each item and keep a weighted progress line at the top.**
    Give every item without subitems, and every subitem, a size in points
    after its label: 1 for a small change, 2 for a typical item, 3 for a
    large one, 5 for one that is nearly too large to review, as in
    `- [ ] **B3.** [3] ...`. Under the title, write
    `Progress: 7 of 47 items, 14 of 89 points (16%)`: the checked items
    against all items, and their points against all points, with the
    percentage of points. Update it with each tick. Re-estimate an item
    when it is split; the numbers are rough.

## Example

```markdown
# Time series epochs: implementation plan

Progress: 0 of 5 items, 0 of 9 points (0%)

Spec: [timeseries-dates-and-plotting-spec.md](timeseries-dates-and-plotting-spec.md)

## Phase A: Epoch on `Times`

- [ ] **A1.** [2] C++: add `Epoch` (count and unit) and an optional epoch on
  `Times`; `to` keeps it, `operator==` compares it.
  Tests: default empty; kept by `to`; equality.
- [ ] **A2.** [2] Save/load the epoch in the version 3 `Times` record.
  Tests: round trip with an epoch, without one, at `as` and `d` resolution.
- [ ] **A3.** Python: `Times(..., epoch=)` and `Times.epoch`.
  - [ ] **A3a.** [2] Accept `str`, `datetime64`, and naive `datetime`.
    Tests: all three agree; default is `None`.
  - [ ] **A3b.** [1] Reject invalid epochs.
    Tests: timezone-aware `datetime` raises; `M` resolution raises.

## Phase B: Building `Times` from dates

- [ ] **B1.** [2] ...
```
