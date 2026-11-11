"""Keep class entries compact while exposing inherited public members."""

from docutils import nodes

MARKER = "sb-inherited-members"


def _add_dropdown(app, what, name, obj, options, lines):
    # The autoclass in the dropdown has inherited members, so it gets no
    # dropdown of its own.
    if what != "class" or options.inherited_members:
        return
    if not getattr(obj, "__module__", "").startswith("stablebear."):
        return
    if name.rsplit(".", 1)[-1] != obj.__name__:
        return

    inherited = {
        member
        for base in obj.__mro__[1:]
        for member in vars(base)
        if not member.startswith("_") and member not in vars(obj)
    }
    if not inherited:
        return

    lines.extend([
        "",
        f".. container:: {MARKER}",
        "",
        "   .. dropdown:: All members, including inherited",
        "",
        f"      .. autoclass:: {name}",
        "         :members:",
        "         :inherited-members:",
        "         :undoc-members:",
        "         :no-index:",
        "",
    ])


def _move_dropdowns_after_members(app, doctree):
    # autodoc puts the docstring, and so the dropdown, before the members.
    for node in list(doctree.findall(nodes.container)):
        if MARKER in node["classes"]:
            parent = node.parent
            parent.remove(node)
            parent.append(node)


def setup(app):
    app.setup_extension("sphinx.ext.autodoc")
    app.setup_extension("sphinx_design")
    # After napoleon, which rewrites the docstring at the default priority.
    app.connect("autodoc-process-docstring", _add_dropdown, priority=900)
    app.connect("doctree-read", _move_dropdowns_after_members)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
