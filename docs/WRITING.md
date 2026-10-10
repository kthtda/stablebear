# Documentation writing manifesto

Stablebear's documentation should help readers understand what they can do,
why they would do it, and what the results mean. Write for mathematically
literate users with relevant background knowledge, without assuming familiarity
with every concept or with stablebear's particular conventions. Use the guidance
for the kind of content being written together with the generally applicable
principles below.

## 1. User-facing

This section concerns user guides, tutorials, and notebooks. Guides explain
concepts, when to use an operation, and how to interpret its results. Tutorials
and notebooks develop a worked example around a question or task. Link to the
API reference for the full contract.

1. **Introduce ideas in the order they are needed.** Establish the objects
   and their relationships before discussing operations on them. Explain the
   purpose of a calculation before presenting its mechanics. Readers should
   know what they are trying to learn, not just which command to run next.

2. **Connect sections instead of making abrupt jumps.** Explain how the next
   step follows from the previous result and serves the larger objective.
   A heading organizes the text but does not supply that connection. Keep
   transitions short and specific to the reasoning.

3. **Explain what a quantity measures.** Connect its definition to its
   interpretation. Explain what larger or smaller values mean, which effects
   contribute, and what information a summary leaves out. Give distinct
   quantities separate explanations when their meanings differ, even if
   their computations look similar.

4. **Use visuals to make comparisons easier.** Choose plots or tables when
   they communicate a pattern more clearly than raw output. Define what
   markers, ranges, and uncertainty displays represent. Keep visual encodings
   consistent and use comparable scales where comparison requires them.
   Identify separate scales when used. Make the important elements stand
   out, and check contrast and legibility in both light and dark modes.
   Choose colors that remain distinguishable for readers with color vision
   deficiencies. Do not rely on color alone to distinguish groups or curves;
   use marker shapes, line styles, or direct labels as well. Different marker
   shapes and line styles often distinguish groups more clearly than a large
   set of contrasting colors. Use them to keep the palette small and choose
   colors with good contrast against the background. Check figures
   with color vision deficiency simulations and in grayscale.

   Match the palette to the data: distinct colors for categories, a
   sequential colormap for ordered values, a diverging colormap around a
   meaningful center, and a cyclic colormap for periodic quantities. For
   numerical scales, prefer perceptually uniform colormaps so equal changes
   in value produce comparable visual changes. Avoid rainbow maps such as
   `jet`, which can create apparent boundaries or hide variation. Sequential
   maps should progress steadily in lightness. Include a labeled colorbar
   when color encodes numerical values. See the color references below.

5. **Keep examples simple and purposeful.** Use the simplest data and setup
   that demonstrate the intended behavior. Add complexity only when it is
   part of the lesson. Choose values that make relevant differences visible.
   Include reproducibility controls when the result depends on randomness.
   Omit arguments and setup that do not help explain the example. Wrap code
   to fit the rendered column; aim for roughly 78 characters per line.

6. **Let the evidence determine the interpretation.** Distinguish an
   observation in an example from a general guarantee. Describe useful
   implications without overstating what a demonstration establishes.
   Do not force a result to support a predetermined narrative.

### 1a. User guide

1. **Organize pages around concepts and uses.** Explain when an operation is
   useful and how it fits with related operations. Use focused examples to
   illustrate the explanation, and link to notebooks for extended analyses.

2. **Present figures independently of plotting code.** Put a short
   introduction or question before the figure. Place the "Show code" dropdown
   immediately after the figure, followed by the interpretation. The
   explanation should make sense without opening the dropdown. Explain
   visual encodings close to the figure without narrating every visible value.

3. **Link operations to their API documentation on first mention.** When
   discussing a stablebear operation, link its first mention in the prose of
   each user guide page to the corresponding API entry. Link to the specific
   function or method so readers can find its parameters and behavior directly.
   A name appearing in a code example does not replace this prose link.

### 1b. Notebooks

Notebooks do not require API links on first mention.

1. **Build tutorials around questions and decisions.** Let readers see why
   a method or parameter choice is useful. When comparison is part of the
   lesson, show the alternatives and connect the choice to the results.
   Less informative outcomes can help explain a decision, but do not add
   an artificial search process to an otherwise straightforward example.

2. **Place explanations around the cell outputs.** Put a short introduction
   or question before the plotting cell, then describe the figure and its
   implications after the output. A specific point that the next figure will
   examine can come before the cell. Explain visual encodings close to the
   output without narrating every visible value.

## 2. API

API documentation is a reference manual, not a user guide. Public docstrings
and reference pages should let readers quickly look up what an operation
accepts, returns, and guarantees. Keep entries brief, direct, and complete.
Motivation, conceptual introductions, workflows, and interpretation of results
belong in user guides and notebooks.

1. **State the contract.** Start with what the object or operation does.
   Document parameters, defaults, accepted inputs, return values, and relevant
   constraints or errors. For array-like inputs and outputs, specify shapes
   and the meaning of their axes where these are part of the contract.

2. **Keep essential requirements in the API entry.** Document supported edge
   cases and restrictions that affect correct use, even when a guide can
   leave them out. Do not make readers follow a guide to discover a requirement.

3. **Include examples only to clarify the contract.** A short example can
   resolve ambiguity about a call or its result. Omit examples that merely
   repeat the signature, and leave worked analyses to guides and notebooks.

## 3. Generally applicable

These principles apply across guides, tutorials, notebooks, API docstrings,
reference pages, and captions. Apply the guidance on examples and figures
wherever those appear; it does not require adding them to every entry.

1. **Make each sentence worth reading.** State something concrete that helps
   the reader understand an operation, interpret a result, or make a choice.
   Remove generic commentary that adds no information. Read the paragraph
   again and ask what the reader gains from it.

2. **Respect what the reader already knows.** Avoid elementary explanations
   of familiar concepts. Focus on what is new in the current context. When
   background is needed, give a brief explanation or link to an appropriate
   introduction. Use the glossary for terminology rather than repeatedly
   interrupting the main discussion with definitions. Avoid calling a step
   "obvious" or "easy"; explain what makes it work.

3. **Use concrete language.** Name the objects, conditions, and actions you
   mean. Prefer a direct explanation of what happens to a vague label for it.
   Use terminology consistently, and avoid casual abbreviations that make
   readers translate between names. Use the established domain terms in
   prose, docstrings, captions, and plot labels: write "point cloud" rather
   than "cloud." Follow the glossary and existing API terminology, and use
   the same term for the same concept throughout a page.
   Avoid direct address such as "you" and "your," and avoid colloquial
   expressions. Name the relevant object or operation, or use a direct
   instruction such as "Pass a query dataset." Keep the wording natural
   and precise rather than adopting an unnecessarily formal tone.
   Prefer ordinary punctuation and complete
   sentences to em dashes and parenthetical detours. Use the same sentence
   structure when describing comparable concepts, so readers can easily
   identify what differs.

4. **Use US English consistently.** Use US spelling and usage throughout
   prose, docstrings, captions, and plot labels, for example "color,"
   "behavior," and "normalize." Preserve the original spelling in quotations,
   publication titles, proper names, and API identifiers.

5. **Keep mathematical prose natural and precise.** Introduce symbols before
   using them and state assumptions where they matter. Integrate notation
   into grammatical sentences rather than placing disconnected expressions
   next to each other. Use standard mathematical terminology and typography.
   A formula should make a statement more precise; the surrounding prose
   should explain its role rather than laboriously read it aloud.
   Punctuate displayed equations as parts of sentences. Keep notation
   consistent, and avoid unnecessary subscripts in mathematical expressions.
   The argument should remain understandable when a reader skims the formulas.

6. **Describe actual behavior.** Check the implementation or a small example
   when a factual claim is uncertain. Distinguish mathematical conventions
   from software behavior. Document what the software does today; omit planned
   features and promises about future behavior.
   Include qualifications when they affect understanding, not merely because
   a remote edge case exists. For common mistakes, explain how to recognize
   the problem and recover from it.

7. **Keep implementation detail relevant to the reader.** Include internal
   details when they change how someone should use an API or interpret a
   result. Otherwise, leave them to developer documentation or omit them.
   Match the level of detail to the purpose of the page rather than listing
   everything known about the implementation.

8. **Give shared concepts one home.** Explain reusable concepts and options
   where they can be understood independently of a single application. Link
   to that explanation instead of repeating it. Guides explain usage and
   reasoning; API references specify the full contract. Use compact tables
   when readers need to compare parameters or alternatives. Readers may land
   in the middle of a page; give enough local context to orient them.

9. **Verify claims against results.** Inspect outputs before describing them,
   and check that the text agrees with what is shown. Recheck affected claims
   when results change, and update the documentation accordingly.

10. **Revise for the reader.** Check for ambiguous references, undefined
    terms, and sentences with more than one plausible interpretation. Check
    that each step follows from information already given. Supply missing
    explanations before polishing the wording, then simplify sentences that
    make the reader hold too many qualifications in mind at once.

11. **Finish the rendered document, not just its source.** Keep prose,
    formulas, code, labels, and saved outputs consistent. Rerun notebooks
    after code changes; preserve their outputs for prose-only edits. Rebuild
    the docs after every documentation edit, check warnings, and inspect
    figures when their content or styling changes. Follow the figure/code
    conventions in [AGENTS.md](../AGENTS.md). Do not require the reader to
    rebuild the docs to review the work.

The review question is: **What will the reader understand or be able to do
after this paragraph, and does the example actually support it?**

Further reading: Donald E. Knuth, Tracy Larrabee, and Paul M. Roberts,
[Mathematical Writing](https://jmlr.csail.mit.edu/reviewing-papers/knuth_mathematical_writing.pdf),
especially §§1, 13, and 34. These notes informed the guidance on mathematical
prose, user manuals, and revision above.

For color choices, see Fabio Crameri, Grace E. Shephard, and Philip J. Heron,
[The misuse of colour in science communication](https://doi.org/10.1038/s41467-020-19160-7)
(2020), and Bang Wong,
[Points of view: Color blindness](https://doi.org/10.1038/nmeth.1618) (2011).
These inform the guidance on colormaps and accessibility above.
